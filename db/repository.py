"""
db/repository.py
================
Data-access layer (CRUD) for Golden Batch Analytics.

All database writes go through these functions so that:
  • app.py / routers never build SQL directly
  • Every write is wrapped in a transaction with rollback on failure
  • Audit log entries are created automatically alongside data writes

Usage
-----
  from db.repository import save_batch_run, save_batches, save_cpp_rankings
  from db.connection import SessionLocal

  with SessionLocal() as db:
      run = save_batch_run(db, filename="batch_data.csv", ...)
      save_batches(db, run.id, batch_df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.models import AuditLog, Batch, BatchRun, CppRanking, GoldenProfileMeta

logger = logging.getLogger(__name__)


# ── Audit helpers ──────────────────────────────────────────────────────────────
def _audit(
    db: Session,
    action: str,
    run_id: Optional[int] = None,
    detail: str = "",
    status: str = "success",
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> None:
    """Append one AuditLog row. Does NOT commit – caller is responsible."""
    db.add(AuditLog(
        run_id=run_id,
        action=action,
        detail=detail,
        status=status,
        ip_address=ip_address,
        user_agent=(user_agent[:500] if user_agent else None),
        duration_ms=duration_ms,
    ))


def audit_request(
    db: Session,
    *,
    method: str,
    path: str,
    status_code: int,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    duration_ms: Optional[int] = None,
    run_id: Optional[int] = None,
) -> None:
    """
    Record a single API request in the audit log.
    Committed immediately (fire-and-forget from middleware).
    """
    action = f"{method}:{path}"[:100]
    log_status = "success" if status_code < 400 else ("warning" if status_code < 500 else "failure")
    detail = f"http_status={status_code} duration_ms={duration_ms}"
    _audit(
        db,
        action=action,
        run_id=run_id,
        detail=detail,
        status=log_status,
        ip_address=ip_address,
        user_agent=user_agent,
        duration_ms=duration_ms,
    )
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# BatchRun
# ══════════════════════════════════════════════════════════════════════════════
def save_batch_run(
    db: Session,
    *,
    filename: str,
    data_format: str,
    n_batches: int,
    n_cpps: int,
    n_time_steps: Optional[int] = None,
    has_yield: bool = False,
    golden_pct_used: Optional[float] = None,
    cpp_vars: Optional[list] = None,
    pipeline_meta: Optional[dict] = None,
    run_name: Optional[str] = None,
) -> BatchRun:
    """
    Insert a new BatchRun row and return it (with id populated).
    Also writes an audit_log entry.
    """
    run = BatchRun(
        filename=filename,
        data_format=data_format,
        n_batches=n_batches,
        n_cpps=n_cpps,
        n_time_steps=n_time_steps,
        has_yield=has_yield,
        golden_pct_used=golden_pct_used,
        cpp_vars=cpp_vars or [],
        pipeline_meta=pipeline_meta or {},
        run_name=run_name,
    )
    db.add(run)
    db.flush()  # populate run.id without committing
    _audit(db, "pipeline_built", run_id=run.id,
           detail=f"file={filename!r} format={data_format} batches={n_batches}")
    db.commit()
    db.refresh(run)
    logger.info("BatchRun saved: id=%s file=%r", run.id, filename)
    return run


def get_all_runs(db: Session) -> list[BatchRun]:
    """Return all batch runs ordered newest-first."""
    return db.query(BatchRun).order_by(BatchRun.created_at.desc()).all()


def get_run_by_id(db: Session, run_id: int) -> Optional[BatchRun]:
    return db.query(BatchRun).filter(BatchRun.id == run_id).first()


def delete_run(db: Session, run_id: int) -> bool:
    """Delete a run and all its child records (cascade). Returns True if found."""
    run = get_run_by_id(db, run_id)
    if not run:
        return False
    _audit(db, "run_deleted", run_id=run_id, detail=f"file={run.filename!r}")
    db.delete(run)
    db.commit()
    logger.info("BatchRun deleted: id=%s", run_id)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Batches
# ══════════════════════════════════════════════════════════════════════════════
def save_batches(db: Session, run_id: int, batch_df: pd.DataFrame) -> int:
    """
    Bulk-insert batch metadata rows from a batch_df DataFrame.
    Expected columns: batch_id, yield, quality, is_golden, duration.
    Returns the number of rows inserted.
    """
    rows = []
    for _, row in batch_df.iterrows():
        rows.append(Batch(
            run_id=run_id,
            batch_id=str(row["batch_id"]),
            yield_value=float(row["yield"]) if pd.notna(row.get("yield")) else None,
            quality=str(row.get("quality", "Unknown")),
            is_golden=bool(row.get("is_golden", False)),
            duration=int(row.get("duration", 0)) if pd.notna(row.get("duration")) else None,
        ))
    db.bulk_save_objects(rows)
    _audit(db, "batches_saved", run_id=run_id, detail=f"{len(rows)} batches")
    db.commit()
    logger.info("Saved %d batches for run_id=%s", len(rows), run_id)
    return len(rows)


def get_batches_for_run(db: Session, run_id: int) -> list[Batch]:
    return (
        db.query(Batch)
        .filter(Batch.run_id == run_id)
        .order_by(Batch.batch_id)
        .all()
    )


# ══════════════════════════════════════════════════════════════════════════════
# CPP Rankings
# ══════════════════════════════════════════════════════════════════════════════
def save_cpp_rankings(db: Session, run_id: int, ranking_df: pd.DataFrame) -> int:
    """
    Bulk-insert CPP ranking rows from the compute_combined_ranking() DataFrame.
    Expected columns: base_variable (or feature), rank, corr_score, tree_score,
                      pca_score, lasso_score, avg_score, method_agreement.
    Returns number of rows inserted.
    """
    # Delete previous rankings for this run (idempotent re-save)
    db.query(CppRanking).filter(CppRanking.run_id == run_id).delete()

    rows = []
    for _, row in ranking_df.iterrows():
        var = str(row.get("base_variable") or row.get("variable") or row.get("feature", ""))
        if not var:
            continue
        rows.append(CppRanking(
            run_id=run_id,
            variable=var,
            rank=int(row.get("rank", 0)),
            corr_score=_safe_float(row.get("corr_score")),
            tree_score=_safe_float(row.get("tree_score")),
            pca_score=_safe_float(row.get("pca_score")),
            lasso_score=_safe_float(row.get("lasso_score")),
            avg_score=_safe_float(row.get("avg_score")),
            method_agreement=int(row["method_agreement"]) if pd.notna(row.get("method_agreement")) else None,
        ))
    db.bulk_save_objects(rows)
    _audit(db, "cpp_rankings_saved", run_id=run_id, detail=f"{len(rows)} CPPs ranked")
    db.commit()
    logger.info("Saved %d CPP rankings for run_id=%s", len(rows), run_id)
    return len(rows)


def get_cpp_rankings_for_run(db: Session, run_id: int) -> list[CppRanking]:
    return (
        db.query(CppRanking)
        .filter(CppRanking.run_id == run_id)
        .order_by(CppRanking.rank)
        .all()
    )


# ══════════════════════════════════════════════════════════════════════════════
# Golden Profile
# ══════════════════════════════════════════════════════════════════════════════
def save_golden_profile(
    db: Session,
    run_id: int,
    golden_profile: dict,
    cpp_vars: list,
) -> GoldenProfileMeta:
    """
    Persist golden profile metadata.
    The full per-timepoint arrays are serialised as JSON (mean, std, upper, lower).
    For large profiles (many CPPs × 100 points), only aggregated stats are stored.
    """
    # Delete existing (idempotent)
    db.query(GoldenProfileMeta).filter(GoldenProfileMeta.run_id == run_id).delete()

    cpp_stats = {}
    for var in cpp_vars:
        gp = golden_profile.get(var)
        if not isinstance(gp, dict):
            continue
        cpp_stats[var] = {
            "mean_avg":  float(np.mean(gp["mean"])),
            "std_avg":   float(np.mean(gp["std"])),
            "upper_max": float(np.max(gp["upper"])),
            "lower_min": float(np.min(gp["lower"])),
        }

    meta = GoldenProfileMeta(
        run_id=run_id,
        n_golden=int(golden_profile.get("n_golden", 0)),
        yield_threshold=None,
        t_crit=_safe_float(golden_profile.get("t_crit")),
        golden_batch_ids=golden_profile.get("batch_ids", []),
        cpp_stats=cpp_stats,
    )
    db.add(meta)
    _audit(db, "golden_profile_saved", run_id=run_id,
           detail=f"n_golden={meta.n_golden} cpps={len(cpp_stats)}")
    db.commit()
    db.refresh(meta)
    logger.info("GoldenProfileMeta saved for run_id=%s", run_id)
    return meta


def get_golden_profile_for_run(db: Session, run_id: int) -> Optional[GoldenProfileMeta]:
    return (
        db.query(GoldenProfileMeta)
        .filter(GoldenProfileMeta.run_id == run_id)
        .first()
    )


# ══════════════════════════════════════════════════════════════════════════════
# Audit log
# ══════════════════════════════════════════════════════════════════════════════
def get_audit_log(db: Session, run_id: Optional[int] = None, limit: int = 100) -> list[AuditLog]:
    q = db.query(AuditLog).order_by(AuditLog.created_at.desc())
    if run_id is not None:
        q = q.filter(AuditLog.run_id == run_id)
    return q.limit(limit).all()


# ── Internal helper ────────────────────────────────────────────────────────────
def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        import math
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None

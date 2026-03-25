"""
File Processing Service – CSV Template Generation
===================================================
Generates downloadable template CSV files so users know
exactly what format to upload.

Templates are generated in-memory and returned as byte strings –
no temp files are written to disk.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd


# ── Step 1 template ────────────────────────────────────────────────────────────

def generate_step1_template() -> bytes:
    """
    Generate the Step 1 CPP data template.

    Layout
    ------
    Rows    → CPPs                  (y-axis); first column = CPP_Name label
    Columns → Batch Serial Numbers  (x-axis)

    Each cell = aggregate CPP value for that batch (e.g. mean over batch duration).
    Include a 'yield' row so the system can identify the outcome variable.
    """
    rng = np.random.default_rng(seed=0)

    batch_ids = [f"BATCH_{i+1:03d}" for i in range(10)]

    cpp_specs = [
        ("temperature",  37.0, 0.5),
        ("pH",            7.2, 0.05),
        ("dissolved_o2", 40.0, 3.0),
        ("agitation",   185.0, 8.0),
        ("pressure",      1.0, 0.02),
        ("feed_rate",    42.0, 3.0),
        ("foam_level",   13.0, 2.0),
        ("conductivity",  6.5, 0.15),
        ("yield",        80.0, 5.0),
    ]

    rows = []
    for cpp_name, mean, std in cpp_specs:
        row = {"CPP_Name": cpp_name}
        for bid in batch_ids:
            row[bid] = round(mean + rng.normal(0, std), 3)
        rows.append(row)

    df = pd.DataFrame(rows)

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


# ── Step 2 template ────────────────────────────────────────────────────────────

def generate_step2_template() -> bytes:
    """
    Generate the Step 2 temperature trajectory template.

    Layout (long format)
    ---------------------
    Column 1: timestamp  – numeric time index (e.g. minutes elapsed)
    Column 2: temperature – process temperature in °C

    Represents one batch's complete temperature profile.
    """
    rng = np.random.default_rng(seed=42)

    # 100 time points, 5-minute intervals → 0 to 495 minutes
    n_points = 100
    timestamps = np.arange(0, n_points * 5, 5)     # 0, 5, 10, …, 495 min

    # Realistic fermentation temperature profile
    # Phase 1 (0–20 %): warm-up                  35 → 37 °C
    # Phase 2 (20–80 %): controlled at 37 °C
    # Phase 3 (80–100 %): cool-down              37 → 36 °C
    t = np.linspace(0, 1, n_points)
    warmup   = np.clip(35.0 + 2.0 * t / 0.2, 35.0, 37.0)
    maintain = np.full(n_points, 37.0)
    cooldown = np.where(t > 0.8, 37.0 - 1.0 * (t - 0.8) / 0.2, 37.0)

    temperature = (
        warmup * (t < 0.2)
        + maintain * ((t >= 0.2) & (t <= 0.8))
        + cooldown * (t > 0.8)
        + rng.normal(0, 0.18, n_points)    # sensor noise
    )
    temperature = np.clip(temperature, 34.0, 42.0)

    df = pd.DataFrame({
        "timestamp":   timestamps,
        "temperature": temperature.round(2),
    })

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

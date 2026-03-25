"""
Locust Load Test – Golden Batch Analytics API
==============================================
Target: 50 concurrent users, mixed read/write traffic.

Usage
-----
    # Install: pip install locust
    # Run interactively (web UI at http://localhost:8089):
    locust -f tests/locustfile.py --host http://localhost:8000

    # Headless CI run – 50 users, 5 users/sec ramp, 2-minute duration:
    locust -f tests/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 5 --run-time 2m \
           --html tests/load_report.html --csv tests/load_results

Scenarios
---------
GoldenBatchAPIUser
    Simulates a real analyst:
      60% – GET /health             (lightweight probe)
      20% – GET /api/v1/step1/template  (CSV download)
      10% – POST /api/v1/step1/upload   (file upload with synthetic CSV)
      10% – POST /api/v1/step2/upload   (trajectory upload)

GoldenBatchHeavyUser
    Simulates repeated upload + confirm cycles (stress test):
      50% – POST /api/v1/step1/upload → /confirm
      50% – POST /api/v1/step2/upload → /confirm
"""

from __future__ import annotations

import io
import random
import time

import numpy as np
import pandas as pd
from locust import HttpUser, TaskSet, between, events, task


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic CSV factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_step1_csv(n_batches: int = 20, n_cpps: int = 6) -> bytes:
    """
    Step 1 format: CPPs in rows, batches in columns.

    CPP_Name  | BATCH_001 | BATCH_002 | …
    temperature|   37.2   |   36.8   | …
    pH         |    7.18  |    7.24  | …
    yield      |   87.2   |   82.1   | …
    """
    batch_ids = [f"BATCH_{i:03d}" for i in range(1, n_batches + 1)]
    cpp_names = [
        "temperature", "pH", "dissolved_oxygen", "agitation_speed",
        "feed_rate", "pressure",
    ][:n_cpps]

    rows = []
    for cpp in cpp_names:
        row = {"CPP_Name": cpp}
        row.update({bid: round(random.gauss(37.0, 0.5), 3) for bid in batch_ids})
        rows.append(row)

    # Add yield row
    yield_row = {"CPP_Name": "yield"}
    yield_row.update({bid: round(random.uniform(70, 95), 2) for bid in batch_ids})
    rows.append(yield_row)

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _make_step2_csv(n_points: int = 100) -> bytes:
    """
    Step 2 format: two columns – timestamp + temperature.
    """
    t = np.linspace(0, 600, n_points)
    temp = 37.0 + np.sin(t / 60) * 0.5 + np.random.normal(0, 0.1, n_points)
    df = pd.DataFrame({"timestamp_min": t, "temperature_c": np.round(temp, 3)})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ─────────────────────────────────────────────────────────────────────────────
# Task Sets
# ─────────────────────────────────────────────────────────────────────────────

class AnalystTasks(TaskSet):
    """
    Mixed read/write workload simulating a pharma analyst browsing the tool.
    """

    @task(6)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(2)
    def download_step1_template(self):
        with self.client.get(
            "/api/v1/step1/template", catch_response=True, name="GET /step1/template"
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Template download failed: {resp.status_code}")

    @task(1)
    def download_step2_template(self):
        with self.client.get(
            "/api/v1/step2/template", catch_response=True, name="GET /step2/template"
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Template download failed: {resp.status_code}")

    @task(1)
    def upload_step1(self):
        csv_bytes = _make_step1_csv(n_batches=random.randint(10, 30))
        files = {"file": ("batch_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        with self.client.post(
            "/api/v1/step1/upload",
            files=files,
            catch_response=True,
            name="POST /step1/upload",
        ) as resp:
            if resp.status_code in (200, 429):  # 429 = rate limited (expected under load)
                resp.success()
            else:
                resp.failure(f"Upload failed: {resp.status_code} {resp.text[:200]}")

    @task(1)
    def upload_step2(self):
        csv_bytes = _make_step2_csv(n_points=random.randint(50, 150))
        files = {"file": ("trajectory.csv", io.BytesIO(csv_bytes), "text/csv")}
        with self.client.post(
            "/api/v1/step2/upload",
            files=files,
            catch_response=True,
            name="POST /step2/upload",
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"Upload failed: {resp.status_code} {resp.text[:200]}")


class HeavyUploadTasks(TaskSet):
    """
    Stress scenario: repeated upload → confirm cycles back-to-back.
    """

    @task(1)
    def upload_and_confirm_step1(self):
        # Upload
        csv_bytes = _make_step1_csv(n_batches=20, n_cpps=6)
        files = {"file": ("stress.csv", io.BytesIO(csv_bytes), "text/csv")}
        with self.client.post(
            "/api/v1/step1/upload", files=files, catch_response=True,
            name="POST /step1/upload [heavy]",
        ) as resp:
            if resp.status_code == 429:
                resp.success()
                return
            if resp.status_code != 200:
                resp.failure(f"Upload failed: {resp.status_code}")
                return
            detection = resp.json()
            resp.success()

        # Confirm – use detected columns
        time.sleep(0.1)
        payload = {
            "cpp_column": detection.get("cpp_column", "CPP_Name"),
            "batch_columns": detection.get("batch_columns", []),
            "yield_row_name": detection.get("yield_row_name"),
        }
        with self.client.post(
            "/api/v1/step1/confirm", json=payload, catch_response=True,
            name="POST /step1/confirm [heavy]",
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"Confirm failed: {resp.status_code} {resp.text[:200]}")

    @task(1)
    def upload_and_confirm_step2(self):
        csv_bytes = _make_step2_csv(n_points=100)
        files = {"file": ("traj.csv", io.BytesIO(csv_bytes), "text/csv")}
        with self.client.post(
            "/api/v1/step2/upload", files=files, catch_response=True,
            name="POST /step2/upload [heavy]",
        ) as resp:
            if resp.status_code == 429:
                resp.success()
                return
            if resp.status_code != 200:
                resp.failure(f"Upload failed: {resp.status_code}")
                return
            detection = resp.json()
            resp.success()

        time.sleep(0.1)
        payload = {
            "time_column": detection.get("time_column", "timestamp_min"),
            "temp_column": detection.get("temp_column", "temperature_c"),
        }
        with self.client.post(
            "/api/v1/step2/confirm", json=payload, catch_response=True,
            name="POST /step2/confirm [heavy]",
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"Confirm failed: {resp.status_code} {resp.text[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# User classes
# ─────────────────────────────────────────────────────────────────────────────

class GoldenBatchAPIUser(HttpUser):
    """
    Simulates a typical analyst: mostly reads, occasional uploads.
    Target: 40 of the 50 concurrent users.
    """
    tasks = [AnalystTasks]
    weight = 4
    wait_time = between(1, 3)   # 1–3 s think time between tasks


class GoldenBatchHeavyUser(HttpUser):
    """
    Simulates a power user hammering upload → confirm.
    Target: 10 of the 50 concurrent users.
    """
    tasks = [HeavyUploadTasks]
    weight = 1
    wait_time = between(0.5, 1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Custom event hooks – print summary thresholds to console
# ─────────────────────────────────────────────────────────────────────────────

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    stats = environment.stats
    total = stats.total
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total requests  : {total.num_requests}")
    print(f"Failures        : {total.num_failures}  "
          f"({100 * total.num_failures / max(total.num_requests, 1):.1f}%)")
    print(f"Median latency  : {total.median_response_time:.0f} ms")
    print(f"95th pct latency: {total.get_response_time_percentile(0.95):.0f} ms")
    print(f"99th pct latency: {total.get_response_time_percentile(0.99):.0f} ms")
    print(f"Peak RPS        : {total.total_rps:.1f}")

    # Fail the run if error rate > 5% or p99 > 3000 ms
    fail_reasons = []
    error_rate = total.num_failures / max(total.num_requests, 1)
    if error_rate > 0.05:
        fail_reasons.append(f"Error rate {error_rate:.1%} > 5%")
    p99 = total.get_response_time_percentile(0.99)
    if p99 > 3000:
        fail_reasons.append(f"p99 latency {p99:.0f}ms > 3000ms")

    if fail_reasons:
        print("\nFAIL – SLA breached:")
        for r in fail_reasons:
            print(f"  • {r}")
        environment.process_exit_code = 1
    else:
        print("\nPASS – all SLAs met.")
    print("=" * 60)

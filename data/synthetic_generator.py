"""
Synthetic Pharma Batch Data Generator
======================================
Simulates a fed-batch fermentation process with 8 CPPs.
Yield is driven primarily by temperature stability, pH control, and dissolved O2.

Data returned:
  batch_df     – summary table: batch_id, yield, duration, quality label
  time_series  – dict { batch_id → DataFrame of (time_index, CPP columns) }
"""

import numpy as np
import pandas as pd

from config import CPP_VARIABLES, N_ALIGNED_POINTS


# ── Internal: profile generators ─────────────────────────────────────────────

def _temperature(rng, duration, t):
    """37 °C target. 25% chance of a temperature excursion."""
    temp = 37.0 + rng.normal(0, 0.25, duration)
    if rng.random() < 0.25:
        s = rng.randint(int(0.20 * duration), int(0.60 * duration))
        l = min(rng.randint(8, 20), duration - s)
        direction = rng.choice([-1, 1]) * rng.uniform(2.5, 5.0)
        # smooth spike using a sine envelope
        spike = direction * np.sin(np.pi * np.arange(l) / l)
        temp[s: s + l] += spike
    return np.clip(temp, 30.0, 46.0)


def _pH(rng, duration, t):
    """pH 7.2 target. Random walk with imperfect control."""
    drift = np.cumsum(rng.normal(0, 0.007, duration))
    control = -0.45 * drift          # controller partially compensates
    pH = 7.2 + drift + control + rng.normal(0, 0.035, duration)
    return np.clip(pH, 6.5, 8.0)


def _dissolved_o2(rng, duration, t):
    """Starts ~80 %, decays as cells grow, stabilises ~35 %. 15% DO-crash risk."""
    do = 45.0 * np.exp(-4.0 * t) + 35.0 + rng.normal(0, 1.8, duration)
    if rng.random() < 0.15:
        s = rng.randint(int(0.30 * duration), int(0.60 * duration))
        l = min(rng.randint(10, 25), duration - s)
        do[s: s + l] -= rng.uniform(20, 40)
    return np.clip(do, 5.0, 100.0)


def _agitation(rng, duration, t):
    """Ramps from 150 to 230 RPM over the batch."""
    return np.clip(150.0 + 80.0 * t + rng.normal(0, 5, duration), 100.0, 300.0)


def _pressure(rng, duration, t):
    """Stable around 1.0 bar with minor oscillation."""
    return 1.0 + 0.04 * np.sin(2 * np.pi * t) + rng.normal(0, 0.018, duration)


def _feed_rate(rng, duration, t):
    """Linear ramp to 50 L/h in first half, then plateau."""
    base = np.where(t < 0.5, 100.0 * t, 50.0)
    return np.clip(base + rng.normal(0, 1.8, duration), 0.0, 60.0)


def _foam_level(rng, duration, t):
    """Peaks mid-batch (bell-shaped)."""
    return np.clip(25.0 * np.sin(np.pi * t) ** 2 + rng.normal(0, 3, duration), 0.0, 80.0)


def _conductivity(rng, duration, t):
    """Slowly increases throughout the batch."""
    return 5.0 + 2.5 * t + rng.normal(0, 0.09, duration)


def _build_profile(rng, duration):
    """Assemble all CPP time-series into one DataFrame."""
    t = np.linspace(0, 1, duration)
    return pd.DataFrame({
        'time_index':   np.arange(duration),
        'temperature':  _temperature(rng, duration, t),
        'pH':           _pH(rng, duration, t),
        'dissolved_o2': _dissolved_o2(rng, duration, t),
        'agitation':    _agitation(rng, duration, t),
        'pressure':     _pressure(rng, duration, t),
        'feed_rate':    _feed_rate(rng, duration, t),
        'foam_level':   _foam_level(rng, duration, t),
        'conductivity': _conductivity(rng, duration, t),
    })


# ── Yield model ───────────────────────────────────────────────────────────────

def _compute_yield(profile: pd.DataFrame) -> float:
    """
    Yield is a function of CPP quality scores:
      - Temperature: penalise mean offset from 37 °C and high variability
      - pH:          penalise mean offset from 7.2 and instability
      - Dissolved O2: penalise if minimum drops below 30 %
      - Agitation:   small reward for staying near 190 RPM
    Base yield = 60 %; maximum theoretical = 98 %.
    """
    # Temperature score (0–1)
    t_mean = profile['temperature'].mean()
    t_std  = profile['temperature'].std()
    temp_score = max(0.0, 1.0 - abs(t_mean - 37.0) * 0.8 - t_std * 0.55)

    # pH score (0–1)
    p_mean = profile['pH'].mean()
    p_std  = profile['pH'].std()
    pH_score = max(0.0, 1.0 - abs(p_mean - 7.2) * 5.0 - p_std * 4.0)

    # DO score (0–1)
    do_min = profile['dissolved_o2'].min()
    do_score = max(0.0, (min(do_min, 45.0) - 5.0) / 40.0)

    # Agitation score (small, 0–1)
    agit_score = max(0.0, 1.0 - abs(profile['agitation'].mean() - 190.0) / 100.0)

    yield_pct = (
        60.0
        + 18.0 * temp_score      # temperature is most impactful
        + 12.0 * pH_score        # pH second most
        +  6.0 * do_score        # DO third
        +  2.0 * agit_score      # agitation minor
        + np.random.normal(0, 1.2)   # measurement noise
    )
    return float(np.clip(yield_pct, 45.0, 98.0))


# ── Public API ────────────────────────────────────────────────────────────────

def generate_batch_data(n_batches: int = 60, random_seed: int = 42):
    """
    Generate the historical batch dataset.

    Returns
    -------
    batch_df : pd.DataFrame
        Columns: batch_id, yield, duration, quality, is_golden
    time_series : dict
        { batch_id : pd.DataFrame with time_index + 8 CPP columns }
    """
    np.random.seed(random_seed)
    rng = np.random.RandomState(random_seed)

    records    = []
    time_series = {}

    for i in range(n_batches):
        batch_id = f"BATCH_{i + 1:03d}"
        duration = int(rng.randint(90, 151))          # variable batch length 90–150 steps

        profile = _build_profile(rng, duration)
        profile.insert(0, 'batch_id', batch_id)

        y = _compute_yield(profile)

        time_series[batch_id] = profile
        records.append({'batch_id': batch_id, 'yield': round(y, 2), 'duration': duration})

    batch_df = pd.DataFrame(records)

    # Label batches by quality tier
    q75 = batch_df['yield'].quantile(0.75)
    q50 = batch_df['yield'].quantile(0.50)
    batch_df['quality'] = batch_df['yield'].apply(
        lambda y: 'Golden' if y >= q75 else ('Good' if y >= q50 else 'Poor')
    )
    batch_df['is_golden'] = batch_df['yield'] >= q75

    return batch_df, time_series


def generate_test_batch(scenario: str = 'temperature_excursion', random_seed: int = 999) -> pd.DataFrame:
    """
    Generate a single test batch for the real-time monitoring demo.

    Scenarios
    ---------
    'normal'                – well-controlled batch (should stay in control)
    'temperature_excursion' – temperature spike at ~35–55 % of batch
    'ph_drift'              – pH drifts from target starting mid-batch
    'do_crash'              – dissolved O2 crashes ~45 % into the batch
    """
    np.random.seed(random_seed)
    rng = np.random.RandomState(random_seed)
    duration = N_ALIGNED_POINTS   # keep test batch at 100 steps for simplicity
    t = np.linspace(0, 1, duration)

    # Start with a well-controlled baseline
    temp  = 37.0 + rng.normal(0, 0.18, duration)
    pH    = 7.2  + rng.normal(0, 0.03, duration)
    do    = 45.0 * np.exp(-4.0 * t) + 35.0 + rng.normal(0, 1.5, duration)
    agit  = 150.0 + 80.0 * t + rng.normal(0, 4, duration)
    pres  = 1.0 + 0.04 * np.sin(2 * np.pi * t) + rng.normal(0, 0.015, duration)
    feed  = np.where(t < 0.5, 100.0 * t, 50.0) + rng.normal(0, 1.5, duration)
    foam  = 25.0 * np.sin(np.pi * t) ** 2 + rng.normal(0, 2.5, duration)
    cond  = 5.0 + 2.5 * t + rng.normal(0, 0.08, duration)

    # Apply the requested deviation scenario
    if scenario == 'temperature_excursion':
        # Smooth temperature spike between 35 % and 55 % of batch
        s, l = int(0.35 * duration), int(0.20 * duration)
        spike = 4.5 * np.sin(np.pi * np.arange(l) / l)
        temp[s: s + l] += spike

    elif scenario == 'ph_drift':
        # pH drifts down after 40 % of batch
        s = int(0.40 * duration)
        pH[s:] += np.linspace(0, -0.85, duration - s)

    elif scenario == 'do_crash':
        # DO drops sharply at 45 % of batch, recovers slowly
        s, l = int(0.45 * duration), int(0.25 * duration)
        do[s: s + l] -= np.linspace(0, 35, l)

    scenario_label = {
        'normal':                'Normal (in-control)',
        'temperature_excursion': 'Temperature Excursion',
        'ph_drift':              'pH Drift',
        'do_crash':              'DO Crash',
    }.get(scenario, scenario)

    return pd.DataFrame({
        'batch_id':    f'TEST – {scenario_label}',
        'time_index':  np.arange(duration),
        'temperature':  np.clip(temp, 30.0, 46.0),
        'pH':           np.clip(pH,   6.5,   8.0),
        'dissolved_o2': np.clip(do,   5.0, 100.0),
        'agitation':    np.clip(agit, 100.0, 300.0),
        'pressure':     np.clip(pres,  0.8,   1.3),
        'feed_rate':    np.clip(feed,  0.0,  60.0),
        'foam_level':   np.clip(foam,  0.0,  80.0),
        'conductivity': cond,
    })

# ── Golden Batch Analytics ── Configuration ──────────────────────────────────

# All 8 simulated Critical Process Parameters for a fermentation process
CPP_VARIABLES = [
    'temperature',   # °C  – most critical, target 37.0
    'pH',            # –    – very critical, target 7.2
    'dissolved_o2',  # %    – critical, should stay > 30 %
    'agitation',     # RPM  – moderate impact, target ~190
    'pressure',      # bar  – low impact, target 1.0
    'feed_rate',     # L/h  – moderate impact
    'foam_level',    # %    – low impact
    'conductivity',  # mS/cm – low impact
]

# Human-readable labels and units
CPP_LABELS = {
    'temperature':  'Temperature (°C)',
    'pH':           'pH',
    'dissolved_o2': 'Dissolved O₂ (%)',
    'agitation':    'Agitation (RPM)',
    'pressure':     'Pressure (bar)',
    'feed_rate':    'Feed Rate (L/h)',
    'foam_level':   'Foam Level (%)',
    'conductivity': 'Conductivity (mS/cm)',
}

# Target (nominal) values for each CPP
CPP_TARGETS = {
    'temperature':  37.0,
    'pH':           7.2,
    'dissolved_o2': 40.0,
    'agitation':    190.0,
    'pressure':     1.0,
    'feed_rate':    40.0,
    'foam_level':   15.0,
    'conductivity': 6.5,
}

# Batch quality colour scheme
QUALITY_COLORS = {
    'Golden': '#FFD700',
    'Good':   '#4CAF50',
    'Poor':   '#F44336',
}

# Number of time points after alignment (normalisation)
N_ALIGNED_POINTS = 100

# Yield percentile threshold to define "golden" batches
YIELD_GOLDEN_PERCENTILE = 0.75   # top 25 %
YIELD_GOOD_PERCENTILE   = 0.50   # top 50 %

# MSPC control-limit confidence levels (empirical percentile)
MSPC_LIMIT_95 = 95
MSPC_LIMIT_99 = 99

# PCA components for MSPC model
MSPC_N_COMPONENTS = 5

# PLS components for BPLS model
PLS_N_COMPONENTS = 5

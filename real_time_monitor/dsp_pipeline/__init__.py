"""DSP pipeline — signal processing, beamforming, and strategy interfaces."""

# Existing exports (unchanged)
from dsp_pipeline.harmonic_mask import apply_harmonic_attenuation
from dsp_pipeline.music_angle import estimate_angle_music
from dsp_pipeline.lcmv_beamformer import lcmv_displacement

# Strategy layer (new)
from dsp_pipeline.strategies import (
    SignalCleanerStrategy,
    VitalSignSeparator,
    VMDRLSCleaner,
    EMDHarmonicCleaner,
    EMDPulseCleaner,
    PassthroughCleaner,
    WPDSeparator,
    SOSFilterSeparator,
    AdaptiveStrategySelector,
)

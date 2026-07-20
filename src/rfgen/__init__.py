"""RF-GenLab: deep generative models for synthetic RF signals and anomaly detection.

Package layout
--------------
- ``rfgen.dsp``     : synthetic RF signal generation, impairments, STFT features.
- ``rfgen.data``    : dataset construction (HDF5) and PyTorch data loading.
- ``rfgen.models``  : the generative models (VAE, RealNVP flow, DDPM) + classifier.
- ``rfgen.train``   : training entry points for each model.
- ``rfgen.eval``    : anomaly-detection, generation quality, and figure scripts.
"""

__version__ = "1.0.0"

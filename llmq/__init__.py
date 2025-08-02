try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development when setuptools-scm hasn't run yet
    __version__ = "0.0.0+unknown"

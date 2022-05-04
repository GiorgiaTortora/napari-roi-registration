
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"





from ._widget import subtract_background, register_rois, process_rois

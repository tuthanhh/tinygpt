import importlib.metadata


try:
    __version__ = importlib.metadata.version("tinygpt")
except importlib.metadata.PackageNotFoundError:
    pass

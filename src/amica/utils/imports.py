"""Soft import helper."""
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType

# A mapping from import name to package name (on PyPI) when the package name
# is different.
_INSTALL_MAPPING: dict[str, str] = {
    "codespell_lib": "codespell",
    "cv2": "opencv-python",
    "parallel": "pyparallel",
    "pytest_cov": "pytest-cov",
    "serial": "pyserial",
    "sklearn": "scikit-learn",
    "sksparse": "scikit-sparse",
}

def import_optional_dependency(
    name: str,
    extra: str = "",
) -> ModuleType:
    """Import an optional dependency.

    If a dependency is missing an ImportError with a nice message will be
    raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.

    Returns
    -------
    module : Module
        The imported module when found.
    """
    package_name = _INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name
    if find_spec(name) is None:
        raise ImportError(
            f"Missing optional dependency '{install_name}'. {extra} Use pip or "
            f"conda to install {install_name}."
        )
    return import_module(name)

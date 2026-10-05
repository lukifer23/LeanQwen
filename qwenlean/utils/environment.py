"""Optional dependency discovery, independent of model loading."""

import importlib.metadata as md
import platform
import subprocess

import psutil

PACKAGES = [
    "qwenlean",
    "mlx",
    "mlx-lm",
    "numpy",
    "transformers",
    "torch",
    "peft",
    "trl",
    "sentence-transformers",
]


def package_versions(packages=PACKAGES):
    versions = {}
    for package in packages:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_manifest(check_mps=False):
    versions = package_versions()
    mps = {"built": None, "available": None}
    if check_mps and versions["torch"] is not None:
        import torch

        mps = {
            "built": torch.backends.mps.is_built(),
            "available": torch.backends.mps.is_available(),
        }
    chip = platform.processor()
    if platform.system() == "Darwin":
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "platform": platform.platform(),
        "macos": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
        "chip": chip,
        "ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes": psutil.virtual_memory().available,
        "packages": versions,
        "mps": mps,
    }

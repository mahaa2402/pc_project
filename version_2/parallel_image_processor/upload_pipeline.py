"""Convert uploaded images to P6 PPM (RGB) and run the MPI parallel_ppm binary."""
from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np


def _read_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("OpenCV could not read the image (unsupported or corrupt file).")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        raise ValueError("Expected 1, 3, or 4 channels; got shape " + str(img.shape))
    return img


def write_ppm_p6_rgb(original_path: str, ppm_path: str) -> None:
    """
    Read any format OpenCV supports, convert to RGB, write binary P6 PPM (max 255).
    Matches the project's C++ PPMHandler (P6, RGB, 255).
    """
    bgr = _read_bgr(original_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid image dimensions.")
    out = Path(ppm_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    payload = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    with open(out, "wb") as f:
        f.write(header)
        f.write(payload)


def run_parallel_ppm(
    parallel_dir: Path,
    ppm_relative_to_parallel: str,
    *,
    np_processes: int | None = None,
    exe_name: str = "parallel_ppm",
) -> tuple[bool, str]:
    """
    Run: mpirun -np P ./parallel_ppm <ppm_relative>
    from parallel_dir (e.g. .../parallel_image_processor).
    Returns (success, message_or_stderr).
    """
    exe = parallel_dir / exe_name
    if not exe.is_file():
        return False, f"Missing executable '{exe}'. Build with: mpicxx -std=c++17 main.cpp ../image_utils.cpp -O2 -o parallel_ppm"
    if not os.access(exe, os.X_OK):
        return False, f"'{exe}' is not executable."

    mpirun = shutil.which("mpirun")
    if not mpirun:
        return False, "mpirun not found in PATH (install Open MPI or MPICH)."

    n = np_processes
    if n is None:
        n = int(os.environ.get("MPI_NP", os.environ.get("PARALLEL_NP", "4")))
    n = max(1, n)

    cmd = [mpirun, "-np", str(n), str(exe), ppm_relative_to_parallel]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(parallel_dir),
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("MPI_TIMEOUT_SEC", "600")),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "MPI pipeline timed out."
    except OSError as e:
        return False, str(e)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
        return False, err[:4000]

    return True, ""

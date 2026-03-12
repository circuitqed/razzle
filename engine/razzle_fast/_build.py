"""
Build script for razzle_fast C shared library.

Usage:
    cd engine && python3 razzle_fast/_build.py

Compiles core.c into a shared library (libcore.so) that Python loads via ctypes.
"""

import os
import subprocess
import sys
import sysconfig

here = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(here, "core.c")
output_path = os.path.join(here, "libcore.so")

def build():
    cmd = [
        "gcc",
        "-shared", "-fPIC",
        "-O3", "-march=native", "-std=c11",
        "-o", output_path,
        source_path,
        "-lm",
    ]
    print(f"Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    print(f"Built successfully: {output_path}")

if __name__ == "__main__":
    build()

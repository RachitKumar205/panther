"""llama.cpp's convert_hf_to_gguf.py, run with its own environment (/opt/venv-convert, from llama.cpp's
requirements at the pinned commit). panther.model runs PANTHER_CONVERT with Panther's Python; in the image
PANTHER_CONVERT is this file, which hands the same arguments to the converter under the right Python."""

import os
import sys

os.execv("/opt/venv-convert/bin/python",
         ["/opt/venv-convert/bin/python", "/opt/llama.cpp/convert_hf_to_gguf.py", *sys.argv[1:]])

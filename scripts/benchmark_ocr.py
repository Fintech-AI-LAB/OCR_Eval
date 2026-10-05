#!/usr/bin/env python3
"""Repository entry point for the standalone reference benchmark and profile tools."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'standalone_ocr_eval'))
from ocr_eval.cli import main


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
AIDS-RPi Root Launcher: Forwards execution to raspberry_pi.rpi_monitor.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raspberry_pi.rpi_monitor import main

if __name__ == "__main__":
    main()

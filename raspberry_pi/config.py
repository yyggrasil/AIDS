"""
Configuration Loader Module for Raspberry Pi 3 Edge Deployment (AIDS-RPi).
Ensures raspberry_pi/.env is discovered and loaded with priority over the root .env.
"""

import os
from dotenv import load_dotenv


def load_rpi_env(override: bool = True):
    """
    Discovers and loads environment variables for the Raspberry Pi 3 module.
    Prioritizes raspberry_pi/.env over the project root .env.
    """
    rpi_dir = os.path.dirname(os.path.abspath(__file__))
    rpi_env = os.path.join(rpi_dir, ".env")
    root_env = os.path.join(os.path.dirname(rpi_dir), ".env")

    # 1. Load root .env first without overriding (as baseline fallback)
    if os.path.exists(root_env):
        load_dotenv(dotenv_path=root_env, override=False)

    # 2. Load raspberry_pi/.env with override=True so RPi 3 settings take precedence
    if os.path.exists(rpi_env):
        load_dotenv(dotenv_path=rpi_env, override=override)
    else:
        # Fallback to standard dotenv walk
        load_dotenv()


# Auto-load on module import
load_rpi_env()

from pathlib import Path
import os

# Get the absolute path to the project root directory
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Define paths relative to the root directory
CONFIG_FILE_PATH = ROOT_DIR / "config/config.yaml"
PARAMS_FILE_PATH = ROOT_DIR / "params.yaml"
SCHEMA_FILE_PATH = ROOT_DIR / "schema.yaml"
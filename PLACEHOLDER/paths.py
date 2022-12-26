from pathlib import Path


SRC_DIR = Path(__file__).absolute().parent
ROOT_DIR = SRC_DIR.parent

DATA_DIR = ROOT_DIR / "data"

RAW_PATH = DATA_DIR / "raw"
PROCESSED_PATH = DATA_DIR / "processed"
ANNOTATIONS_PATH = DATA_DIR / "annotations"
ARTIFACTS_PATH = DATA_DIR / "artifacts"
OUTPUT_PATH = DATA_DIR / "output"

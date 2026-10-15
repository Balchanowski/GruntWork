from pathlib import Path

# ścieżki
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESOURCES_DIR = DATA_DIR / "resources"

# pasek surowców
REF_H = 1367
BAR_HEIGHT = 0.05
MIN_SCORE = 0.5
RESOURCE_NAMES = ["gold", "lumber", "oil", "hunger"]
OCR_HEIGHT = 64

# minimapa
MINIMAP_REF = (63, 72, 367, 367)
MINIMAP_MARGIN = 1

DEBUG = False
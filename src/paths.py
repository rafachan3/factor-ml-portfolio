"""Project path constants."""

from pathlib import Path

# Repository root (parent of src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MMA_SAMPLE_PATH = DATA_DIR / "mma_sample_v2.csv"
FACTOR_LIST_PATH = DATA_DIR / "factor_char_list.csv"
MKT_IND_PATH = DATA_DIR / "mkt_ind.csv"

TARGET_COLUMN = "stock_exret"
EXPECTED_START = "2000-01"
EXPECTED_END = "2023-12"
N_EXPECTED_FACTORS = 147

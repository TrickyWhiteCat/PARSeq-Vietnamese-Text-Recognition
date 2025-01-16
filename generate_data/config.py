from pathlib import Path
import os

# CWD
CWD = Path(__file__).parent.resolve()
os.chdir(CWD)
# Main

NUM_FILES = 100_000
NUM_PROCESSORS = 14
GENERATED_DIR = Path("D:/parseq_gen/generated")
LMDB_DIR = Path(r"D:/parseq_gen/lmdb")

LABEL_DIR = GENERATED_DIR / "labels"
IMAGES_DIR = GENERATED_DIR / "imgs"

COMBINED_LABEL_FILE = GENERATED_DIR / 'label.txt'

TRAIN_LABEL_FILE = GENERATED_DIR / 'train_label.txt'
TRAIN_LMDB_PATH = LMDB_DIR / 'train'
TEST_LABEL_FILE = GENERATED_DIR / 'test_label.txt'
TEST_LMDB_PATH = LMDB_DIR / 'test'

TEST_SPLIT_SIZE=0.95

LMDB_REDUNDANCY_RATIO = 0.3

# Resources
RESOUCRE_DIR = Path('./resources')
DICTIONARY_PATH = RESOUCRE_DIR / "dictionary.txt"
DICTIONARY_SEPERATOR = ','
FONT_DIR = RESOUCRE_DIR / 'fonts'
BACKGROUND_DIR = RESOUCRE_DIR / 'backgrounds'
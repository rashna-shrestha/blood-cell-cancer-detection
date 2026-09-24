#all the settings for this project live here, predict.py reads them from this file

from pathlib import Path
BASE_PATH = Path(__file__).resolve().parent
MODEL_PATH = BASE_PATH/"Cell_xception_blanked_19_0.983.keras"
INPUT_SIZE = 299

BLANK_CORNER = True
CORNER_TOP = 0.88   
CORNER_LEFT = 0.72  

CLASSES = ["Benign", "Early Pre-B", "Pre-B", "Pro-B"]

SEG_SIZE = (192, 256)
MIN_STAINED_PCT = 1.0
MAX_STAINED_PCT = 40.0

LEARNING_RATE = 0.01
SIZE_INNER = 200     #nodes in the hidden Dense layer
DROPRATE = 0.2       #20% of those nodes switched off during training
EPOCHS = 20          #best epoch was 19
BATCH_SIZE = 32
SEED = 42

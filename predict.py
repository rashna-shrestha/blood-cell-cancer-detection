"""Classify a blood cell image and count its stained cells.

usage: python predict.py <image_path> [<image_path> ...]
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   #hide TensorFlow's startup messages

import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

import config

#take GPU memory only as needed, so this can run next to a notebook
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

#load the model once, then reuse it for every image
model = keras.models.load_model(config.MODEL_PATH)


def read_image(image_path):
    #reading bytes then decoding also works for Windows paths with non-English characters
    img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read an image from {image_path}")
    return img


def predict(image_path):
    img = read_image(image_path)

    #same steps as the training images: resize, blank the corner, BGR -> RGB, scale to [-1, 1]
    img = cv2.resize(img, (config.INPUT_SIZE, config.INPUT_SIZE), interpolation=cv2.INTER_AREA)
    if config.BLANK_CORNER:
        h, w = img.shape[:2]
        img[int(h * config.CORNER_TOP):, int(w * config.CORNER_LEFT):] = 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = preprocess_input(np.array([img], dtype="float32"))

    #the model outputs logits, softmax turns them into probabilities that add up to 1
    logits = model.predict(X, verbose=0)[0]
    probabilities = tf.nn.softmax(logits).numpy()
    return dict(zip(config.CLASSES, probabilities))


def make_mask(rgb):
    #from the notebook: white where the purple stain is, black everywhere else
    a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1]
    a = cv2.GaussianBlur(a, (5, 5), 0)
    _, m = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)    #drop specks
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)   #fill pinholes
    return m


def find_cells(image_path):
    img = read_image(image_path)

    #same steps the notebook used for data/seg: blank the corner, then shrink to 192x256
    h, w = img.shape[:2]
    img[int(h * config.CORNER_TOP):, int(w * config.CORNER_LEFT):] = 255
    small = cv2.resize(img, config.SEG_SIZE[::-1], interpolation=cv2.INTER_AREA)
    mask = make_mask(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    regions = cv2.connectedComponents(mask)[0] - 1   #separate white blobs, minus the background
    stained_pct = (mask > 0).mean() * 100

    warning = None
    if stained_pct < config.MIN_STAINED_PCT:
        warning = f"only {stained_pct:.1f}% of the image is stained, very few stained cells found"
    elif stained_pct > config.MAX_STAINED_PCT:
        warning = f"{stained_pct:.1f}% of the image is stained, is this really a blood smear?"

    return {"mask": mask, "regions": regions, "stained_pct": stained_pct, "warning": warning}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python predict.py <image_path> [<image_path> ...]")

    for path in sys.argv[1:]:
        scores = predict(path)
        cells = find_cells(path)

        print(path)
        print("  prediction:", max(scores, key=scores.get))
        for name, p in scores.items():
            print(f"  {name:<12} {p:.4f}")
        print(f"  stained regions: {cells['regions']} ({cells['stained_pct']:.1f}% of the image)")
        if cells["warning"]:
            print("  WARNING:", cells["warning"])

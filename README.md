# Blood Cell Cancer Classification

A deep learning project for classifying blood smear images and identifying stained regions in microscopy samples. The pipeline combines a pretrained **Xception** model with computer-vision-based segmentation to provide both **cell-class prediction** and **visual stain analysis**.

## Overview

This project supports:

* Blood cell image classification using TensorFlow/Keras
* Stained-region segmentation using **CIELAB color analysis** and **Otsu thresholding**
* An interactive **Streamlit** interface for image upload and analysis
* A **FastAPI** backend for API-based inference
* A command-line prediction script for local or batch inference
* Docker-based deployment for the API

The trained model classifies images into four categories:

* Benign
* Early Pre-B
* Pre-B
* Pro-B

---

## Project Structure

```text
.
├── app.py
├── api.py
├── predict.py
├── config.py
├── requirements.txt
├── Dockerfile
├── Cell_xception_blanked_19_0.983.keras
├── notebook.ipynb
├── .gitignore
├── .dockerignore
└── README.md
```

### File Description

| File                                   | Description                                |
| -------------------------------------- | ------------------------------------------ |
| `app.py`                               | Streamlit dashboard for image analysis     |
| `api.py`                               | FastAPI prediction API                     |
| `predict.py`                           | Command-line inference script              |
| `config.py`                            | Model and preprocessing configuration      |
| `notebook.ipynb`                       | Model development and experimentation      |
| `requirements.txt`                     | Python dependencies                        |
| `Dockerfile`                           | Container configuration for API deployment |
| `Cell_xception_blanked_19_0.983.keras` | Trained Keras model                        |

---

## Key Features

* Upload blood smear images through a browser interface
* Classify samples using an Xception-based deep learning model
* Detect stained cell regions using LAB color-space segmentation
* Reduce segmentation noise using morphological image processing
* Count connected stained regions using connected-component analysis
* Display class probabilities for each prediction
* Calculate the percentage of the image covered by detected stain
* Flag samples with unusually low or high stain coverage
* Run the project locally through Streamlit or FastAPI
* Containerize the API using Docker

---

## Tech Stack

* Python 3.12
* TensorFlow / Keras
* Xception
* OpenCV
* Streamlit
* FastAPI
* Uvicorn
* NumPy

---

## Model Details

The model configuration is defined in `config.py`.

### Model

* **Architecture:** Xception
* **Input size:** `299 × 299`
* **Model file:** `Cell_xception_blanked_19_0.983.keras`
* **Number of classes:** 4

### Classes

```text
Benign
Early Pre-B
Pre-B
Pro-B
```

## Prediction Pipeline

The prediction pipeline follows these steps:

```text
Input Blood Smear Image
          ↓
Resize to 299 × 299
          ↓
Corner Blanking
          ↓
BGR → RGB
          ↓
Xception Preprocessing
          ↓
Xception Model
          ↓
Softmax Probabilities
          ↓
Predicted Class
```

### Steps

1. Resize the input image to the required dimensions.
2. Apply the configured corner-blanking step.
3. Convert the image from BGR to RGB.
4. Apply the preprocessing required by the Xception model.
5. Run the image through the trained model.
6. Apply softmax to obtain class probabilities.
7. Return the predicted class and probabilities.

---

## Stained-Region Segmentation

In addition to classification, the project includes a computer-vision pipeline for analyzing stained regions within the microscopy image.

```text
Input Image
     ↓
BGR → LAB
     ↓
Extract a Channel
     ↓
Gaussian Blur
     ↓
Otsu Thresholding
     ↓
Binary Mask
     ↓
Morphological Processing
     ↓
Connected Components
     ↓
Stained Region Analysis
```

### Segmentation Steps

1. Convert the image to the **LAB color space**.
2. Extract the `a` channel to isolate relevant stain information.
3. Apply Gaussian blur to reduce image noise.
4. Apply Otsu thresholding to create a binary mask.
5. Perform morphological opening and closing.
6. Detect connected components.
7. Count detected stained regions.
8. Calculate the percentage of the image covered by stain.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rashna-shrestha/blood-cell-cancer-detection.git
cd blood-cell-cancer-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Streamlit Application

Run:

```bash
streamlit run app.py
```

This launches the interactive blood cell analysis interface in your browser.

### FastAPI Server

Run:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The interactive API documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

### Command-Line Inference

Run:

```bash
python predict.py path/to/image.jpg
```

The script provides:

* Predicted class
* Per-class probabilities
* Number of detected stained regions
* Percentage of image covered by stain
* Stain-coverage warnings

---

## API Usage

### Health Check

```bash
curl http://127.0.0.1:8000/
```

### Prediction Endpoint

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

### Example Response

```json
{
  "prediction": "Pre-B",
  "probabilities": {
    "Benign": 0.0321,
    "Early Pre-B": 0.1245,
    "Pre-B": 0.7312,
    "Pro-B": 0.1122
  },
  "stained_regions": 18,
  "stained_pct": 12.5,
  "warning": null
}
```

---

## Docker

### Build the Image

```bash
docker build -t blood-cell-classifier .
```

### Run the Container

```bash
docker run -p 8000:8000 blood-cell-classifier
```

The FastAPI service will be exposed on port `8000`.

---

## Configuration

Main configuration values are defined in `config.py`.

Important parameters include:

| Parameter         | Purpose                                         |
| ----------------- | ----------------------------------------------- |
| `MODEL_PATH`      | Location of the trained model                   |
| `INPUT_SIZE`      | Image dimensions expected by the model          |
| `BLANK_CORNER`    | Controls the corner-blanking preprocessing step |
| `CLASSES`         | Classification output labels                    |
| `SEG_SIZE`        | Working size used for segmentation              |
| `MIN_STAINED_PCT` | Minimum expected stain coverage                 |
| `MAX_STAINED_PCT` | Maximum expected stain coverage                 |

---

## Results

The trained Xception model achieved approximately **98.3% validation accuracy** during the experiments.

The segmentation pipeline additionally provides quantitative information about detected stained regions, including the number of connected regions and the percentage of the image covered by detected stain.

> Validation performance depends on the dataset, data split, preprocessing, and training configuration. It should not be interpreted as clinical diagnostic accuracy.

---

## Notes

* The project is designed for microscopy-based blood smear images.
* The classification model and preprocessing pipeline are tuned to the dataset used during development.
* The segmentation pipeline is intended to provide additional visual and quantitative stain analysis.
* Input images with significantly different imaging conditions may produce different results.

---

## Disclaimer

This project is developed for **educational and research purposes**. It is not intended to provide medical diagnosis or replace evaluation by qualified healthcare professionals.

---

## Author

**Rashna Shrestha**

Bachelor of Information Technology

GitHub: `rashna-shrestha`

"""A very simple web API for the blood cell classifier.

run locally:   uvicorn api:app
then open:     http://127.0.0.1:8000/docs   (upload an image and click Execute)
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

import config
import predict   #importing this loads the model once, when the server starts

app = FastAPI(title="Blood Cell Classifier")


@app.get("/")
def health():
    return {"status": "ok", "model": config.MODEL_PATH.name}


@app.post("/predict")
def classify(file: UploadFile):
    #predict.py works with file paths, so save the upload to a temporary file first
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "upload"
        path.write_bytes(file.file.read())
        try:
            scores = predict.predict(path)
            cells = predict.find_cells(path)
        except ValueError:
            raise HTTPException(status_code=400, detail="could not read the uploaded file as an image")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "prediction": ranked[0][0],
        #float() because numpy numbers cannot be turned into JSON
        "probabilities": {name: round(float(p), 4) for name, p in ranked},
        "stained_regions": cells["regions"],
        "stained_pct": round(float(cells["stained_pct"]), 2),
        "warning": cells["warning"],
    }

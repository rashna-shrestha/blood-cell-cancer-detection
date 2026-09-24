FROM python:3.12-slim

WORKDIR /app

#install the libraries first: docker caches this step, so changing the code later does not reinstall them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy only what the API needs, not the notebook or the data folder
COPY config.py predict.py api.py ./
COPY Cell_xception_blanked_19_0.983.keras ./

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# MLOps Mini Project – Iris Classification

API de prédiction de fleurs Iris avec pipeline CI/CD complète.

## Stack
- Python 3.11 / scikit-learn / FastAPI
- Docker
- GitHub Actions

## Lancer en local

```bash
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

## Tester l'API

```bash
# Santé
curl http://localhost:8000/health

# Prédiction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Docker

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

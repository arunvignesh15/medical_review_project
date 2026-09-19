# Medicine Review Sentiment Analysis

## Project Structure

```
Medicine_Review_Project
│
├── data
├── models
├── src
├── static
├── templates
├── app.py
├── train.py
├── requirements.txt
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Dataset

Copy the dataset into

```
data/
```

Example

```
data/
    medicine_reviews.csv
```

---

## Train Model

```bash
python train.py
```

Generated files

```
models/

    random_forest.pkl

    tfidf_vectorizer.pkl
```

---

## Run FastAPI

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000
```

---

## Technologies

- Python
- NLP
- TF-IDF
- Random Forest
- FastAPI
- Bootstrap
- Joblib

---

## Workflow

```
Review

↓

Preprocessing

↓

TF-IDF

↓

Random Forest

↓

Prediction

↓

FastAPI UI
```

We deployed in AWS  and Azure

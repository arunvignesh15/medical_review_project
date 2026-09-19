import os

import pandas as pd

from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_text
from src.feature_engineering import (
    create_tfidf,
    transform_text,
    save_vectorizer
)
from src.model_training import *


DATA_PATH = "data/medicine_reviews.csv"

MODEL_FOLDER = "models"

os.makedirs(MODEL_FOLDER, exist_ok=True)


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Change these column names if your dataset differs
TEXT_COLUMN = "review"
TARGET_COLUMN = "label"

df = df[[TEXT_COLUMN, TARGET_COLUMN]]

df.dropna(inplace=True)

print("Cleaning reviews...")

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(

    df[TEXT_COLUMN],
    df[TARGET_COLUMN],
    test_size=0.2,
    random_state=42

)

print("Creating TF-IDF...")

vectorizer, X_train = create_tfidf(X_train)

X_test = transform_text(vectorizer, X_test)

print("Training Random Forest...")

model = train_model(X_train, y_train)

evaluate_model(model, X_test, y_test)

print("Saving model...")

save_model(model, "models/random_forest.pkl")

save_vectorizer(vectorizer, "models/tfidf_vectorizer.pkl" )

print("\nTraining Complete.")
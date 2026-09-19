import joblib
from src.preprocessing import preprocess_text


def predict_review(review):

    model = joblib.load("models/random_forest.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    review = preprocess_text(review)

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    probability = model.predict_proba(review_vector).max()

    return prediction, probability
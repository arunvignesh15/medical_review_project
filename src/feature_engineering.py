from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def create_tfidf(X_train):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    return vectorizer, X_train_tfidf


def transform_text(vectorizer, X):

    return vectorizer.transform(X)


def save_vectorizer(vectorizer, path):

    joblib.dump(vectorizer, path)


def load_vectorizer(path):

    return joblib.load(path)
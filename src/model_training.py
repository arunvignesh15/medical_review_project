import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    prediction = model.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, prediction))

    print(classification_report(y_test, prediction))


def save_model(model, path):

    joblib.dump(model, path)


def load_model(path):

    return joblib.load(path)
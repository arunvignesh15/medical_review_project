import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download only once
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Clean a medicine review.

    Steps
    -----
    1. Lowercase
    2. Remove HTML tags
    3. Remove URLs
    4. Remove numbers
    5. Remove punctuation
    6. Remove stopwords
    7. Lemmatization
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove html
    text = re.sub(r"<.*?>", " ", text)

    # Remove url
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove digits
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)
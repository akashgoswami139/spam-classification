from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


def load_artifacts():
    """Load the trained SVM model and TF-IDF vectorizer."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found: {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_message(message: str) -> dict:
    """Predict whether an SMS message is ham or spam."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    message = message.strip()

    if not message:
        raise ValueError("message cannot be empty")

    model, vectorizer = load_artifacts()

    message_tfidf = vectorizer.transform([message])
    prediction = int(model.predict(message_tfidf)[0])

    # LinearSVC does not provide predict_proba().
    decision_score = float(model.decision_function(message_tfidf)[0])

    if prediction == 1:
        label = "Spam"
        explanation = "The model classified this message as spam."
    else:
        label = "Ham"
        explanation = "The model classified this message as a legitimate message."

    return {
        "prediction": prediction,
        "label": label,
        "decision_score": decision_score,
        "explanation": explanation,
    }


if __name__ == "__main__":
    test_message = "Congratulations! You have won a free prize. Call now!"
    result = predict_message(test_message)
    print(result)

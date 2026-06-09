from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# ==============================
# LOAD MODELS
# ==============================
logistic_model = joblib.load("models/logistic.pkl")
svm_model = joblib.load("models/svm.pkl")
nb_model = joblib.load("models/naive_bayes.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ==============================
# MODEL METRICS
# ==============================
model_metrics = {
    "logistic": {
        "name": "Logistic",
        "accuracy": 88,
        "confusion_matrix": [[4321, 640], [475, 4564]]
    },
    "svm": {
        "name": "SVM",
        "accuracy": 87,
        "confusion_matrix": [[4293, 668], [555, 4484]]
    },
    "naive_bayes": {
        "name": "Naive Bayes",
        "accuracy": 85,
        "confusion_matrix": [[4214, 747], [729, 4310]]
    }
}

# ==============================
# HELPER FUNCTION
# ==============================
def predict_model(model, text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]

    if hasattr(model, "predict_proba"):
        prob = max(model.predict_proba(text_vector)[0]) * 100
    else:
        prob = 75.0

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "sentiment": sentiment,
        "confidence": round(prob, 2)
    }

# ==============================
# GRAPH (OPTIONAL)
# ==============================
def generate_graph(comparison):
    models = [item["model"] for item in comparison]
    confidences = [item["confidence"] for item in comparison]

    plt.figure()
    plt.bar(models, confidences)
    plt.xlabel("Models")
    plt.ylabel("Confidence (%)")
    plt.title("Model Comparison")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode()

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return "Backend is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    review = data.get("review")
    model_choice = data.get("model")

    if not review:
        return jsonify({"error": "No review provided"}), 400

    try:
        # ==============================
        # SINGLE MODEL (🔥 FIXED)
        # ==============================
        if model_choice in model_metrics:
            if model_choice == "logistic":
                model = logistic_model
            elif model_choice == "svm":
                model = svm_model
            else:
                model = nb_model

            result = predict_model(model, review)

            return jsonify({
                **result,
                "accuracy": model_metrics[model_choice]["accuracy"],
                "confusion_matrix": model_metrics[model_choice]["confusion_matrix"]
            })

        # ==============================
        # ALL MODELS
        # ==============================
        elif model_choice == "all":
            comparison = []

            for key in model_metrics:
                if key == "logistic":
                    model = logistic_model
                elif key == "svm":
                    model = svm_model
                else:
                    model = nb_model

                comparison.append({
                    "model": model_metrics[key]["name"],
                    **predict_model(model, review),
                    "accuracy": model_metrics[key]["accuracy"],
                    "confusion_matrix": model_metrics[key]["confusion_matrix"]
                })

            graph = generate_graph(comparison)

            return jsonify({
                "comparison": comparison,
                "graph": graph
            })

        else:
            return jsonify({"error": "Invalid model selected"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)

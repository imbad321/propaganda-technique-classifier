from flask import Flask, jsonify, render_template, request

import inference
from labels import LABEL_DESCRIPTIONS

app = Flask(__name__)
inference.load_model()


@app.route("/")
def index():
    return render_template("index.html", labels=LABEL_DESCRIPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Request body must include non-empty 'text'."}), 400

    results = inference.predict(text)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)

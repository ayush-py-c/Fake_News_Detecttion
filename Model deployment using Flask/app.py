from flask import Flask, jsonify, render_template, request
import joblib
import re
import string
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)


def _resolve_model_path():
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    candidates = [
        os.path.join(parent_dir, "model.pkl"),
        os.path.join(parent_dir, "Model.pkl"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Could not find model.pkl or Model.pkl in project root")


model_path = _resolve_model_path()
Model = joblib.load(model_path)

HIGH_RISK_PATTERN = re.compile(
    r"\b(died|dead|death|killed|assassinated|passed\s+away|rip|breaking)\b",
    re.IGNORECASE,
)

@app.route('/')
def index():
    return render_template("index.html", result=None, error=None, input_text="")

def wordpre(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)  # remove special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    wants_json = request.is_json
    if wants_json:
        payload = request.get_json(silent=True) or {}
        txt = payload.get('text', '')
    else:
        txt = request.form.get('txt', '')

    if not txt or not str(txt).strip():
        if wants_json:
            return jsonify({"error": "Text is required"}), 400
        return render_template("index.html", result=None, error="Text is required", input_text=str(txt))

    clean_txt = wordpre(str(txt))
    clean_series = pd.Series([clean_txt])

    try:
        pred = Model.predict(clean_series)
        value = int(pred[0])
        confidence = None
        true_prob = None
        fake_prob = None
        if hasattr(Model, "predict_proba"):
            probs = Model.predict_proba(clean_series)
            confidence = float(probs[0][value])
            if probs.shape[1] >= 2:
                fake_prob = float(probs[0][0])
                true_prob = float(probs[0][1])
    except Exception as exc:
        if wants_json:
            return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500
        return render_template("index.html", result=None, error=f"Prediction failed: {str(exc)}", input_text=str(txt))

    is_high_risk_claim = bool(HIGH_RISK_PATTERN.search(clean_txt))
    label = "Fake News" if value == 0 else "True News"
    label_class = "fake" if value == 0 else "true"
    reason = None

    # Safety override: avoid asserting "True" on rumor-like or low-margin sensitive claims.
    if value == 1 and (is_high_risk_claim or (true_prob is not None and true_prob < 0.90)):
        label = "Needs Verification"
        label_class = "verify"
        reason = "Sensitive or uncertain claim. Verify with trusted news sources."

    response = {
        "prediction": value,
        "label": label,
        "label_class": label_class,
        "confidence": round(confidence * 100, 2) if confidence is not None else None,
        "cleaned_text": clean_txt,
        "reason": reason,
        "scores": {
            "fake": round(fake_prob * 100, 2) if fake_prob is not None else None,
            "true": round(true_prob * 100, 2) if true_prob is not None else None,
        },
    }

    if wants_json:
        return jsonify(response)
    return render_template("index.html", result=response, error=None, input_text=str(txt))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

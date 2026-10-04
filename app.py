"""Flask app. Train first with python -m src.train, then run python app.py."""
import json
import os
import pickle
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parent
MAX_CHARACTERS = 50000


def create_app(model_path=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
    path = Path(model_path) if model_path else ROOT / "models" / "best_pipeline.pkl"
    model = None
    if path.exists():
        # Only deserialize artifacts produced locally by src.train.
        with path.open("rb") as stream:
            model = pickle.load(stream)
    manifest_path = ROOT / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    model_name = manifest.get("best_model", "Trained classifier").replace("_", " ").title()

    def predict(title, article):
        if not isinstance(title, str) or not isinstance(article, str):
            raise ValueError("Title and article must be text strings.")
        if len(title) > 300:
            raise ValueError("Please keep the title to 300 characters or fewer.")
        if len(article) > MAX_CHARACTERS:
            raise ValueError("Please keep the article to 50,000 characters or fewer.")
        article = clean_text(article)
        if len(re.findall(r"\b\w+\b", article)) < 10:
            raise ValueError("Please enter an article with at least 10 words.")
        if model is None:
            raise RuntimeError("The model is not ready. Run python -m src.train first.")
        label = int(model.predict([clean_text(title) + "\n" + article])[0])
        return {"label": "Potentially fake" if label else "Genuine", "class_id": label,
                "model": model_name,
                "note": "This prediction reflects text patterns in the training dataset. It does not verify facts."}

    @app.route("/", methods=["GET", "POST"])
    def index():
        title = request.form.get("title", "")
        article = request.form.get("article", "")
        result, error, status = None, None, 200
        if request.method == "POST":
            try:
                result = predict(title, article)
            except ValueError as exc:
                error, status = str(exc), 400
            except RuntimeError as exc:
                error, status = str(exc), 503
        return render_template("index.html", title=title, article=article, result=result, error=error,
                               model_name=model_name, ready=model is not None), status

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object containing article and an optional title."), 400
        try:
            return jsonify(predict(payload.get("title", ""), payload.get("article", "")))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        if request.path.startswith("/api/"):
            return jsonify(error="The submitted article is too large."), 413
        return render_template("index.html", title="", article="", result=None,
                               error="The submitted article is too large.", model_name=model_name,
                               ready=model is not None), 413

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)

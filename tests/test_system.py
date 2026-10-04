"""Integration checks: serialization parity, held-out split integrity and Flask routes."""
import json
import pickle
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

from app import create_app
from src.data import ROOT
from src.train import make_pipeline


class FlaskIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "tests" / f"test_pipeline_{uuid.uuid4().hex}.pkl"
        genuine = "Officials published the annual budget report with audited figures and public records."
        fake = "SHOCKING secret conspiracy exposed!!! Everyone must share this unbelievable hidden story now!"
        cls.articles = [genuine + f" Meeting number {i}." for i in range(8)] + [
            fake + f" Rumor number {i}!" for i in range(8)]
        cls.model = make_pipeline(LinearSVC(random_state=42), k=30)
        cls.model.fit(cls.articles, [0] * 8 + [1] * 8)
        with cls.path.open("wb") as stream:
            pickle.dump(cls.model, stream)
        cls.client = create_app(cls.path).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)

    def test_pickle_round_trip_predictions(self):
        with self.path.open("rb") as stream:
            restored = pickle.load(stream)
        np.testing.assert_array_equal(self.model.predict(self.articles), restored.predict(self.articles))

    def test_browser_submission_and_escaping(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.client.post("/", data={"article": self.articles[0], "title": "<script>alert(1)</script>"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"MODEL PREDICTION", response.data)
        self.assertNotIn(b"<script>alert(1)</script>", response.data)

    def test_api_matches_pipeline(self):
        text = self.articles[0]
        response = self.client.post("/api/predict", json={"article": text})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["class_id"], int(self.model.predict([text])[0]))
        self.assertNotIn("confidence", response.json)

    def test_invalid_inputs(self):
        for payload in ({}, {"article": "   "}, {"article": 12}, {"article": "word " * 11000},
                        {"article": self.articles[0], "title": "a" * 301}):
            with self.subTest(payload_type=type(payload.get("article"))):
                self.assertEqual(self.client.post("/api/predict", json=payload).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json=[]).status_code, 400)
        self.assertEqual(self.client.post("/", data={"article": " "}).status_code, 400)

    def test_request_size_limit(self):
        self.assertEqual(self.client.post("/api/predict", data=b"x" * 300000,
                                         content_type="application/json").status_code, 413)

    def test_missing_model(self):
        client = create_app(self.path.with_name("missing_model_for_test.pkl")).test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.post("/api/predict", json={"article": self.articles[0]}).status_code, 503)


class TrainedArtifactTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "models" / "manifest.json").exists(), "Train models first")
    def test_split_integrity_and_deployment(self):
        splits = pd.read_csv(ROOT / "reports" / "split_manifest.csv")
        for column in ("content_hash", "prefix_hash", "source_row"):
            self.assertEqual(int(splits.groupby(column).split.nunique().max()), 1)
        manifest = json.loads((ROOT / "models" / "manifest.json").read_text())
        validation = pd.read_csv(ROOT / "reports" / "validation_comparison.csv")
        self.assertEqual(manifest["best_model"], validation.loc[validation.macro_f1.idxmax(), "model"])
        with (ROOT / "models" / "best_pipeline.pkl").open("rb") as stream:
            best = pickle.load(stream)
        with (ROOT / "models" / manifest["models"][manifest["best_model"]]["file"]).open("rb") as stream:
            original = pickle.load(stream)
        sample = pd.read_csv(ROOT / "reports" / "test_articles.csv", nrows=10).article
        np.testing.assert_array_equal(best.predict(sample), original.predict(sample))
        response = create_app().test_client().post("/api/predict", json={"article": sample.iloc[0]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["class_id"], int(best.predict([sample.iloc[0]])[0]))


if __name__ == "__main__":
    unittest.main()

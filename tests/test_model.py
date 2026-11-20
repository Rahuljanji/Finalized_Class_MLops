from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

MODEL_PATH = Path("models/model.pkl")
FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
VALID_CLASSES = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Champion model not found at {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


def test_model_pipeline_structure(model):
    """Ensure the champion is a Pipeline with Scaler and Model stages."""
    assert hasattr(model, "steps"), "Model should be a scikit-learn Pipeline"
    step_names = [name for name, _ in model.steps]
    assert "Scaler" in step_names, "Pipeline missing 'Scaler' stage"
    assert "Model" in step_names, "Pipeline missing 'Model' stage"


def test_single_row_inference(model):
    """Test inference on typical flower measurements."""
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=FEATURES,
    )
    pred = model.predict(sample)
    assert len(pred) == 1
    assert pred[0] in VALID_CLASSES
    assert pred[0] == "Iris-setosa"


def test_batch_inference_and_probabilities(model):
    """Test batch inference and verify probability distributions sum to ~1.0."""
    batch = pd.DataFrame([
        [5.0, 3.6, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3],
        [6.5, 3.0, 5.5, 1.8],
    ], columns=FEATURES)

    preds = model.predict(batch)
    assert len(preds) == 3
    for p in preds:
        assert p in VALID_CLASSES

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(batch)
        assert probas.shape == (3, 3)
        for row in probas:
            assert np.isclose(np.sum(row), 1.0, atol=1e-5)


def test_prediction_determinism(model):
    """Ensure identical inputs yield identical outputs (deterministic)."""
    sample = pd.DataFrame([[6.0, 3.0, 4.5, 1.5]], columns=FEATURES)
    pred_1 = model.predict(sample)[0]
    pred_2 = model.predict(sample)[0]
    assert pred_1 == pred_2

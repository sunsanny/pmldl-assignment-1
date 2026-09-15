"""
Stage 2: Model Engineering
Feature engineering -> training -> evaluation -> packaging.

"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_PATH = Path("metrics.json")

# Categorical features encoded with one-hot
CATEGORICAL_FEATURES = ["Series", "Court", "Surface", "Round"]

# Numeric features, including the engineered differences
NUMERIC_FEATURES = ["Rank_1", "Rank_2", "Pts_1", "Pts_2", "Best of", "rank_diff", "pts_diff"]

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 12

MLFLOW_EXPERIMENT = "atp_match_prediction"


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    print(f"[load] train: {len(train_df)} lines, test: {len(test_df)} lines")
    return train_df, test_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features and drops the ones the model must not use.

    Player names are dropped on purpose: there are ~1200 unique players,
    one-hot encoding them would produce thousands of sparse columns.
    The player's strength is already captured by rank and points.
    """
    df = df.copy()

    # The difference between players is far more informative than raw values:
    # a negative rank_diff means Player_1 is ranked higher (lower number = better).
    df["rank_diff"] = df["Rank_1"] - df["Rank_2"]
    df["pts_diff"] = df["Pts_1"] - df["Pts_2"]

    return df


def build_pipeline() -> Pipeline:
    """
    Builds a preprocessing + model pipeline.

    Wrapping everything into a single Pipeline means the API will apply
    exactly the same transformations at prediction time - no risk of
    train/serve skew.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes classification metrics on the test set."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("[eval] Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    return metrics


def main() -> None:
    train_df, test_df = load_processed_data()

    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train_df[feature_columns], train_df["target"]
    X_test, y_test = test_df[feature_columns], test_df["target"]

    # Local file-based tracking - no separate MLflow server needed,
    # which keeps the automated pipeline simple to run.
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        print("[train] Model trained")

        metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "n_train_samples": len(X_train),
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model")

    # Packaging: the API container loads exactly this file.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "model.pkl")
    print(f"[save] Model saved to {MODELS_DIR / 'model.pkl'}")

    # Metrics as a separate file so DVC can track them between runs
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("[done] Stage 2 finished")


if __name__ == "__main__":
    main()
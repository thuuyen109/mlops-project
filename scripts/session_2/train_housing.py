import logging
import os
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow

mlflow.set_tracking_uri("http://localhost:8080")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    filename="logs/train_housing.log",        # đường dẫn file log
    filemode="w",                     # 'w' = ghi đè, 'a' = append
)

logger = logging.getLogger("housing")
logger.addHandler(logging.StreamHandler())


def train():
    model_name = "housing_price_model"
    _version = "3"
    _max_iter = 2000
    _tol = 1e-4
    _random_state = 24

    mlflow.set_experiment(model_name)
    # Paths
    PROJECT_ROOT = Path(os.getcwd())
    DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
    ARTIFACT_DIR = PROJECT_ROOT / "scripts" / "session_1"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = ARTIFACT_DIR / "housing_linear.joblib"

    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Artifact dir: {ARTIFACT_DIR}")
    import pandas as pd

    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    logger.info("Preparing features and target...")
    # Identify target and basic features from the CSV header
    TARGET = "Price"
    NUM_FEATURES = [
        "Avg. Area Income",
        "Avg. Area House Age",
        "Avg. Area Number of Rooms",
        "Avg. Area Number of Bedrooms",
        "Area Population",
    ]

    X = df[NUM_FEATURES]
    y = df[TARGET]

    logger.info("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=_random_state
    )

    logger.info("Building pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            # ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                SGDRegressor(
                    max_iter=_max_iter,
                    tol=_tol,
                    learning_rate="optimal",
                    random_state=_random_state,
                    verbose=1,
                ),
            ),
        ]
    )

    logger.info("Training model...")
    with mlflow.start_run(run_name=f"{model_name}_v{_version}") as run:
        mlflow.log_param( "max_iter", _max_iter)
        mlflow.log_param( "tol", _tol)
        mlflow.log_param( "learning_rate", "optimal")
        mlflow.log_param( "random_state", _random_state)
        model.fit(X_train, y_train)

        # Evaluate model performance
        logger.info("Evaluating model performance...")

        # Make predictions on training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for training set
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Calculate metrics for test set
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Log training metrics
        logger.info("Training set metrics:")
        logger.info(f"  MSE: {train_mse:.4f}")
        logger.info(f"  MAE: {train_mae:.4f}")
        logger.info(f"  R²: {train_r2:.4f}")

        # Log test metrics
        logger.info("Test set metrics:")
        logger.info(f"  MSE: {test_mse:.4f}")
        logger.info(f"  MAE: {test_mae:.4f}")
        logger.info(f"  R²: {test_r2:.4f}")

        # Log model performance summary
        logger.info("Model performance summary:")
        logger.info(f"  Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        logger.info(
            f"  Training RMSE: {train_mse**0.5:.4f}, Test RMSE: {test_mse**0.5:.4f}"
        )

        # save the model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to: {MODEL_PATH}")
        mlflow.log_metric(
            "train_mse", train_mse)
        mlflow.log_metric(
            "train_mae", train_mae)
        mlflow.log_metric(
            "train_r2", train_r2)
        mlflow.log_metric(
            "test_mse", test_mse)
        mlflow.log_metric(
            "test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_artifact(MODEL_PATH, "artifacts")
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    train()

import mlflow

import pandas as pd
import numpy as np
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    filename="logs/train_iris.log",       
    filemode="w",                     
)

logger = logging.getLogger("iris")
logger.addHandler(logging.StreamHandler())

# MLflow tracking URI (bạn đổi theo cổng của server bạn đang chạy)
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("iris_classification")


def train():

    logger.info("Loading Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(np.c_[iris["data"], iris["target"]], columns=iris.feature_names + ["target"])
    X = df[["petal length (cm)", "petal width (cm)"]].to_numpy()
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Các tổ hợp params để thử
    param_grid = [
        {"C": 0.1, "penalty": "l2", "solver": "lbfgs"},
        {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        {"C": 1.0, "penalty": "l1", "solver": "liblinear"},
        {"C": 0.5, "penalty": "elasticnet", "l1_ratio": 0.3, "solver": "saga"},
    ]

    for params in param_grid:
        with mlflow.start_run():
            model = LogisticRegression(max_iter=300, **params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Log toàn bộ tham số và metric
            mlflow.log_params(params)
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
            mlflow.sklearn.log_model(model, "model")

            logger.info("✅ Model performance:")
            logger.info(f"Accuracy: {acc:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    train()

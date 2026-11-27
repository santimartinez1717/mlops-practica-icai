import argparse
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Servidor MLflow: se toma de la variable de entorno MLFLOW_TRACKING_URI.
# Si no está definida, se usa el servidor MLflow de DagsHub del proyecto.
# Las credenciales se leen de MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD.
DAGSHUB_MLFLOW_URI = "https://dagshub.com/santimartinez1717/mlops-practica-icai.mlflow"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DAGSHUB_MLFLOW_URI))

parser = argparse.ArgumentParser(description="Entrena un Random Forest sobre Iris")
parser.add_argument("--n-estimators", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=None)
args = parser.parse_args()

# Cargar el conjunto de datos
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Iniciar un experimento de MLflow
with mlflow.start_run():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, 'model.pkl')

    # MLflow >= 3.x serializa con skops: el modelo lo hemos creado nosotros,
    # así que declaramos como confiable el tipo interno de los árboles.
    mlflow.sklearn.log_model(
        model,
        name="random-forest-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print(f"Experimento registrado con MLflow en {mlflow.get_tracking_uri()}")

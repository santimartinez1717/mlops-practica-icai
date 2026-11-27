# Proyecto de Clasificación del Conjunto de Datos Iris

Este repositorio contiene un proyecto de ejemplo de **Machine Learning** que utiliza un modelo de **Random Forest Classifier** para clasificar el famoso conjunto de datos de **Iris**. El proyecto incluye el código para entrenar el modelo, las dependencias necesarias y un archivo de datos de ejemplo.

## Ficheros del Repositorio

A continuación se describen los archivos principales incluidos en este repositorio:

* **`train.py`**: Este script de Python carga el conjunto de datos Iris, lo divide en conjuntos de entrenamiento y prueba, entrena un modelo **Random Forest** y lo guarda como `model.pkl` y registra la ejecución en MLflow.
* **`requirements.txt`**: Contiene todas las bibliotecas de Python necesarias para ejecutar el proyecto, incluyendo `scikit-learn`, `pandas`, `numpy`, `dvc` y `mlflow`.
* **`data/iris_dataset.csv.dvc`**: Puntero DVC al conjunto de datos de Iris, almacenado en el remoto de DagsHub.

---

## Requisitos

Para ejecutar este proyecto, necesitas tener **Python 3.x** instalado. Se recomienda usar un entorno virtual para gestionar las dependencias.

Puedes instalar las bibliotecas necesarias usando el siguiente comando:

```bash
pip install -r requirements.txt
```

## Datos (DVC + DagsHub)

El dataset `data/iris_dataset.csv` está versionado con DVC y se guarda en el remoto S3 de DagsHub:

```bash
dvc pull
```

## Entrenamiento y seguimiento con MLflow

`train.py` registra cada ejecución (parámetros, `accuracy` y el modelo) en el servidor MLflow indicado por la variable `MLFLOW_TRACKING_URI`. Si no se define, usa el servidor MLflow de DagsHub del proyecto, que requiere credenciales:

```bash
export MLFLOW_TRACKING_USERNAME=<usuario-dagshub>
export MLFLOW_TRACKING_PASSWORD=<token-dagshub>

python train.py                                  # n_estimators=100
python train.py --n-estimators 200
python train.py --n-estimators 50 --max-depth 3
```

Para usar un servidor MLflow local:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python train.py
```

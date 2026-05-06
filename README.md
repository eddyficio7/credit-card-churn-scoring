# Credit Card Churn Scoring

Proyecto de clasificación binaria para predecir **churn de clientes de tarjeta de crédito** a partir de variables demográficas, de relación con el banco y de comportamiento transaccional.

El proyecto forma parte del **Proyecto Final del Módulo 1** del diplomado **Artificial Intelligence & Large Language Models for Financial Markets**. El objetivo es construir una solución de machine learning de punta a punta: desde el análisis exploratorio y validación de datos, hasta una aplicación interactiva de inferencia desplegable con Docker.

---

## Autor:

**Edgar Cano Orozco**
GitHub: `eddyficio7`

---

## Descripción general

El proyecto construye un sistema de scoring para estimar la probabilidad de abandono de clientes de tarjeta de crédito.

El flujo completo incluye:

- análisis exploratorio de datos;
- creación de variables derivadas;
- validación de datos con `Pandera`;
- serialización eficiente en formato Parquet;
- entrenamiento de modelos clásicos con `scikit-learn`;
- entrenamiento de una red neuronal MLP con `PyTorch`;
- tracking de experimentos con `MLflow`;
- empaquetamiento del código en estructura profesional tipo `src-layout`;
- pruebas automatizadas con `pytest`;
- control de calidad con `ruff` y `pre-commit`;
- aplicación interactiva en `Streamlit`;
- comparación entre modelo champion y modelo challenger;
- ejecución reproducible con `Docker`;
- versionado de datos y modelos con `DVC`.

---

## Problema

El objetivo es predecir si un cliente de tarjeta de crédito abandonará la relación con el banco.

Se trata de un problema de **clasificación binaria supervisada**, donde la variable objetivo es:

- `0` = `Existing Customer`
- `1` = `Attrited Customer`

Dado que el churn representa la clase minoritaria y tiene relevancia de negocio, el proyecto da especial atención a métricas como:

- `recall`;
- `F1-score`;
- `ROC AUC`;

y no solamente a `accuracy`.

---

## Dataset

Se utiliza el dataset **Credit Card Customers** de Kaggle.

El archivo original utilizado en el proyecto es:

```text
data/raw/BankChurners.csv
```

El dataset contiene variables relacionadas con:

- perfil demográfico del cliente;
- antigüedad y relación con el banco;
- tipo de tarjeta;
- uso de productos;
- comportamiento transaccional;
- actividad reciente;
- utilización del crédito.

Durante el EDA se eliminaron columnas irrelevantes para modelado y se construyeron variables derivadas para enriquecer la señal predictiva.

### Columnas eliminadas

Se eliminaron:

- `CLIENTNUM`, por ser un identificador del cliente;
- columnas de Naive Bayes incluidas en el dataset original, ya que no representan variables reales del cliente para construir un modelo propio.

### Variables derivadas creadas

Se construyeron tres variables adicionales:

- `Avg_Trans_Amt` = `Total_Trans_Amt / Total_Trans_Ct`
- `Products_Per_Month` = `Total_Relationship_Count / Months_on_book`
- `Contacts_Per_Inactive_Month` = `Contacts_Count_12_mon / (Months_Inactive_12_mon + 1)`

Estas variables buscan capturar:

- monto promedio por transacción;
- profundidad de relación ajustada por antigüedad;
- frecuencia de contacto ajustada por inactividad.

---

## Objetivo del proyecto

Construir un sistema de scoring de churn que permita:

- estimar la probabilidad de abandono de un cliente;
- clasificar el riesgo de churn;
- comparar un modelo clásico de machine learning contra una red neuronal;
- servir predicciones desde una aplicación interactiva en Streamlit;
- empaquetar el proyecto en un contenedor Docker;
- versionar datos y modelos con DVC;
- demostrar buenas prácticas de ingeniería de software para proyectos de machine learning.

---

## Arquitectura general

El proyecto separa la lógica exploratoria de la lógica productiva.

Los notebooks se utilizan para análisis, experimentación y documentación del proceso, mientras que el código reutilizable vive dentro de `src/churn_scoring`.

El flujo general es:

```text
data/raw
   ↓
limpieza + feature engineering
   ↓
validación con Pandera
   ↓
data/interim y data/processed en Parquet
   ↓
entrenamiento de modelos
   ↓
serialización de artefactos
   ↓
inferencia con Streamlit
   ↓
ejecución reproducible con Docker
```

La aplicación utiliza como modelo principal un pipeline de **Gradient Boosting** y muestra como comparación experimental un modelo **MLP en PyTorch**.

---

## Estructura del proyecto

```text
credit-card-churn-scoring/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── gradient_boosting_tuned_pipeline.joblib
│   ├── churn_mlp_state_dict.pt
│   └── pytorch_preprocessor.joblib
├── notebooks/
│   ├── 01_eda_credit_card_churn.ipynb
│   ├── 02_pipeline_scikit_learn.ipynb
│   ├── 03_validation_tracking_churn.ipynb
│   └── 04_pytorch_autograd_mlp.ipynb
├── scripts/
│   ├── build_churn_base.py
│   └── train_pytorch_churn_model.py
├── src/
│   └── churn_scoring/
│       ├── app/
│       │   ├── inference.py
│       │   └── streamlit_app.py
│       ├── data/
│       │   ├── loader.py
│       │   └── validation.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── features/
│       │   └── engineering.py
│       ├── models/
│       │   ├── base.py
│       │   ├── factory.py
│       │   ├── sklearn_model.py
│       │   └── torch_model.py
│       └── utils/
│           └── logging.py
├── tests/
├── Dockerfile
├── .dockerignore
├── .dvcignore
├── .pre-commit-config.yaml
├── .gitignore
├── .env.example
├── dvc.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Flujo de trabajo

El proyecto sigue un flujo de trabajo end-to-end:

1. Carga del dataset original `BankChurners.csv`.
2. Eliminación de columnas no aptas para modelado.
3. Construcción de variables derivadas.
4. Validación del dataset con `Pandera`.
5. Serialización del dataset validado en formato Parquet.
6. Entrenamiento de modelos:
   - pipeline clásico con `scikit-learn`;
   - red neuronal MLP con `PyTorch`.
7. Evaluación de modelos con métricas de clasificación binaria.
8. Serialización de modelos y preprocesadores.
9. Construcción de app de inferencia en Streamlit.
10. Empaquetado con Docker.
11. Versionado de datos y modelos con DVC.

---

## Validación de datos

La validación de datos se realiza con `Pandera`.

El schema valida, entre otros elementos:

- tipos de datos esperados;
- rangos válidos en variables numéricas;
- ausencia de valores nulos en columnas críticas;
- etiquetas válidas de la variable objetivo;
- consistencia entre `Credit_Limit`, `Total_Revolving_Bal` y `Avg_Open_To_Buy`;
- consistencia de la variable derivada `Avg_Trans_Amt`;
- ausencia de filas duplicadas.

El objetivo es evitar que datos corruptos o inconsistentes lleguen al entrenamiento o a la inferencia.

---

## Modelos entrenados

### Modelos clásicos

Se compararon tres modelos clásicos de machine learning:

- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`

La comparación se realizó con `StratifiedKFold` y múltiples métricas:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`

El mejor modelo clásico fue `GradientBoostingClassifier`, posteriormente ajustado con `RandomizedSearchCV`.

### Modelo PyTorch

También se entrenó un modelo tipo **Multilayer Perceptron (MLP)** usando PyTorch.

Este modelo utiliza un preprocesador separado para transformar variables numéricas y categóricas antes de pasar los datos a la red neuronal.

El MLP se incluye como modelo challenger para comparar un enfoque de deep learning contra el pipeline clásico.

### Modelo champion y challenger

| Rol | Modelo | Uso en la app |
|---|---|---|
| Champion | Gradient Boosting Tuned Pipeline | Resultado principal |
| Challenger | PyTorch MLP | Comparación experimental |

El modelo Gradient Boosting se usa como decisión principal debido a su mejor desempeño global en evaluación. El modelo PyTorch se muestra como comparación experimental, ya que sus probabilidades pueden diferir por diferencias de arquitectura, calibración y preprocesamiento.

---

## Resultados

### Modelo clásico base vs modelo tuneado

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting Base | 0.9625 | 0.9560 | 0.8031 | 0.8729 | 0.9894 |
| Gradient Boosting Tuned | **0.9674** | **0.9544** | **0.8369** | **0.8918** | **0.9918** |

### Matriz de confusión del modelo tuneado

- Verdaderos negativos: **1688**
- Falsos positivos: **13**
- Falsos negativos: **53**
- Verdaderos positivos: **272**

### Resultado del modelo PyTorch MLP

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| PyTorch MLP | 0.9462 | 0.8401 | 0.8221 | 0.8310 | 0.9787 |

El modelo PyTorch obtuvo un desempeño competitivo, aunque inferior al Gradient Boosting tuneado. Por esta razón se utiliza como modelo challenger y no como modelo principal de decisión.

---

## Interpretación del modelo clásico

El análisis de importancia de variables mostró que el modelo final se apoya principalmente en variables de comportamiento transaccional y profundidad de relación con el banco.

### Variables más importantes

- `Total_Trans_Ct`
- `Total_Revolving_Bal`
- `Total_Trans_Amt`
- `Total_Ct_Chng_Q4_Q1`
- `Total_Relationship_Count`

También aportaron señal útil algunas variables derivadas creadas durante el EDA, como:

- `Avg_Trans_Amt`
- `Products_Per_Month`
- `Contacts_Per_Inactive_Month`

En contraste, las variables demográficas tuvieron una importancia relativamente baja.

---

## Aplicación Streamlit

La app permite capturar información de un cliente y obtener:

- probabilidad estimada de churn;
- nivel de riesgo;
- clasificación esperada;
- recomendación de retención;
- comparación experimental entre Gradient Boosting y PyTorch MLP.

El resultado principal corresponde al modelo **Gradient Boosting Tuned Pipeline**, considerado como modelo champion.

El modelo **PyTorch MLP** se muestra en una sección expandible como comparación experimental.

---

## Instalación local

Clonar el repositorio:

```bash
git clone https://github.com/eddyficio7/credit-card-churn-scoring.git
cd credit-card-churn-scoring
```

Instalar dependencias:

```bash
uv sync --extra dev
```

Verificar que el paquete sea importable:

```bash
uv run python -c "import churn_scoring; print(churn_scoring.__file__)"
```

---

## Uso del proyecto

### Construir dataset base y validado

```bash
uv run python scripts/build_churn_base.py
```

Este script genera:

```text
data/interim/churn_base.parquet
data/processed/churn_validated.parquet
```

### Entrenar modelo PyTorch

```bash
uv run python scripts/train_pytorch_churn_model.py
```

Este script genera:

```text
models/churn_mlp_state_dict.pt
models/pytorch_preprocessor.joblib
```

### Ejecutar app Streamlit

```bash
uv run streamlit run src/churn_scoring/app/streamlit_app.py
```

La app estará disponible en:

```text
http://localhost:8501
```

---

## Ejecución con Docker

Construir la imagen:

```bash
docker build -t credit-card-churn-scoring .
```

Ejecutar el contenedor:

```bash
docker run --rm -p 8501:8501 credit-card-churn-scoring
```

Abrir la aplicación en:

```text
http://localhost:8501
```

---

## Versionado de datos y modelos con DVC

Los datos y modelos pesados se controlan con DVC, no directamente con Git.

Artefactos versionados:

```text
data/raw/BankChurners.csv
data/interim/churn_base.parquet
data/processed/churn_validated.parquet
models/gradient_boosting_tuned_pipeline.joblib
models/churn_mlp_state_dict.pt
models/pytorch_preprocessor.joblib
```

Para recuperar artefactos desde el remote configurado:

```bash
uv run dvc pull
```

Para enviar artefactos al remote:

```bash
uv run dvc push
```

En este proyecto se configuró un remote local para fines de reproducibilidad durante el desarrollo.

---

## Tracking de experimentos con MLflow

El proyecto utiliza `MLflow` para registrar experimentos de entrenamiento, métricas y artefactos.

Para abrir la interfaz local de MLflow:

```bash
uv run mlflow ui --backend-store-uri mlflow.db
```

o, si se utiliza el directorio local de runs:

```bash
uv run mlflow ui --backend-store-uri mlruns
```

La UI permite comparar ejecuciones, revisar métricas y analizar los artefactos generados durante el entrenamiento.

---

## Calidad de código y pruebas

### Linting

```bash
uv run ruff check src tests scripts
```

### Formato

```bash
uv run ruff format src tests scripts
```

### Tests

```bash
uv run pytest
```

Estado actual de pruebas:

```text
46 passed
```

El proyecto cuenta con pruebas para:

- carga y guardado de datasets;
- feature engineering;
- validación con Pandera;
- métricas de clasificación;
- wrappers de modelos;
- factory de modelos;
- capa de inferencia de la app.

---

## Diseño de software

El proyecto aplica separación de responsabilidades y patrones de diseño.

### Separación de responsabilidades

- `data/`: carga y validación de datos.
- `features/`: creación de variables derivadas.
- `models/`: wrappers de modelos e interfaces comunes.
- `evaluation/`: métricas de evaluación.
- `app/`: lógica de inferencia y aplicación Streamlit.
- `scripts/`: ejecución de flujos de construcción y entrenamiento.

### Strategy / Factory

Se definió una interfaz común `BaseChurnModel`, implementada por:

- `SklearnChurnModel`
- `TorchChurnModel`

Además, se implementó una factory para crear modelos según el tipo requerido. Esto permite intercambiar modelos sin modificar la lógica principal de la aplicación.

---

## Artefactos principales

| Artefacto | Descripción |
|---|---|
| `models/gradient_boosting_tuned_pipeline.joblib` | Pipeline scikit-learn tuneado |
| `models/churn_mlp_state_dict.pt` | Pesos del modelo PyTorch MLP |
| `models/pytorch_preprocessor.joblib` | Preprocesador usado por PyTorch |
| `data/interim/churn_base.parquet` | Dataset base con features derivadas |
| `data/processed/churn_validated.parquet` | Dataset validado con Pandera |

---

## Consideraciones sobre probabilidades

El modelo Gradient Boosting y el modelo PyTorch MLP pueden producir probabilidades distintas para un mismo cliente.

Esto puede ocurrir por varias razones:

- son modelos con arquitecturas diferentes;
- utilizan procesos de entrenamiento distintos;
- las probabilidades no necesariamente están calibradas de la misma forma;
- el modelo Gradient Boosting tuvo mejor desempeño global en evaluación.

Por esta razón, la app utiliza Gradient Boosting como modelo champion y muestra PyTorch MLP como comparación experimental.

---

## Conclusiones

El proyecto demuestra un flujo completo de machine learning aplicado a un problema de churn de clientes de tarjeta de crédito.

El modelo Gradient Boosting tuneado mostró el mejor desempeño global y se utiliza como modelo champion dentro de la app. El modelo PyTorch MLP se integró como challenger para comparar un enfoque de deep learning contra un pipeline clásico.

Más allá del desempeño predictivo, el proyecto enfatiza prácticas de ingeniería de software para machine learning:

- estructura profesional del repositorio;
- validación de datos;
- pruebas automatizadas;
- tracking de experimentos;
- versionado de artefactos;
- app interactiva;
- despliegue reproducible con Docker.

---

## Trabajo futuro

Algunas mejoras posibles son:

- calibrar probabilidades de ambos modelos;
- agregar explicabilidad local con SHAP;
- mejorar la arquitectura del MLP;
- registrar modelos en MLflow Model Registry;

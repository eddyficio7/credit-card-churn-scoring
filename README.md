# Credit Card Churn Scoring

Proyecto de clasificación binaria para predecir **churn de clientes de tarjeta de crédito** a partir de variables demográficas, de relación con el banco y de comportamiento transaccional.

El proyecto forma parte del **Proyecto Final del Módulo 1** del diplomado **Artificial Intelligence & Large Language Models for Financial Markets**. El objetivo es construir una solución de machine learning de punta a punta: desde el análisis exploratorio y modelado, hasta una app de inferencia y despliegue reproducible.

---

## Autor

**Edd**
GitHub: `<tu_usuario_github>`

---

## Estado actual del proyecto

Actualmente el proyecto ya cuenta con:

- definición funcional del problema;
- estructura inicial del repositorio con enfoque profesional;
- análisis exploratorio de datos (EDA);
- creación de variables derivadas;
- pipeline clásico con `scikit-learn` usando `ColumnTransformer` y `Pipeline`;
- comparación de 3 modelos con validación cruzada;
- tuning del mejor modelo;
- evaluación final sobre conjunto holdout;
- serialización del pipeline con `joblib`.

---

## Problema

El objetivo es predecir si un cliente de tarjeta de crédito abandonará la relación con el banco.

Se trata de un problema de **clasificación binaria supervisada**, donde la variable objetivo es:

- `0` = `Existing Customer`
- `1` = `Attrited Customer`

Dado que el churn suele representar la clase minoritaria y además tiene relevancia de negocio, el proyecto da especial atención a métricas como **recall**, **F1-score** y **ROC AUC**, no solamente a `accuracy`.

---

## Dataset

Se utiliza el dataset **Credit Card Customers** de Kaggle.

El dataset contiene variables relacionadas con:

- perfil del cliente;
- antigüedad y relación con el banco;
- tipo de tarjeta;
- uso de productos;
- comportamiento transaccional;
- actividad reciente.

Durante el EDA se eliminaron columnas irrelevantes para modelado y se construyeron variables derivadas para enriquecer la señal predictiva.

### Variables derivadas creadas

- `Avg_Trans_Amt` = `Total_Trans_Amt / Total_Trans_Ct`
- `Products_Per_Month` = `Total_Relationship_Count / Months_on_book`
- `Contacts_Per_Inactive_Month` = `Contacts_Count_12_mon / (Months_Inactive_12_mon + 1)`

---

## Objetivo del proyecto

Construir un sistema de scoring de churn que permita:

- estimar la probabilidad de abandono de un cliente;
- clasificar el riesgo de churn;
- comparar enfoques de modelado clásico y deep learning;
- servir como base para una app interactiva en Streamlit;
- avanzar hacia un entregable reproducible y desplegable.

---

## Flujo de trabajo actual

Hasta este punto, el pipeline clásico sigue este flujo:

1. carga del dataset procesado desde el notebook de EDA;
2. preparación de `X` e `y`;
3. separación train/test con `train_test_split` estratificado;
4. identificación de variables numéricas y categóricas;
5. preprocesamiento con `ColumnTransformer`:
   - imputación de numéricas con mediana;
   - escalado con `StandardScaler`;
   - imputación de categóricas con la moda;
   - codificación con `OneHotEncoder`;
6. comparación de modelos con validación cruzada;
7. tuning del mejor modelo con `RandomizedSearchCV`;
8. evaluación final en holdout;
9. serialización del pipeline final con `joblib`.

---

## Modelos comparados

En la etapa clásica se compararon tres modelos:

- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`

La comparación se realizó con `StratifiedKFold` y múltiples métricas:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`

---

## Mejor modelo clásico

El mejor modelo base fue **GradientBoostingClassifier**, por obtener el mejor desempeño global en validación cruzada, especialmente en:

- `F1-score`
- `Recall`
- `ROC AUC`

Posteriormente, este modelo fue ajustado mediante `RandomizedSearchCV`.

### Resultado del tuning

El mejor modelo tuneado mejoró el desempeño respecto al modelo base, especialmente en **recall** y **F1-score**, manteniendo una **precision** muy alta.

---

## Resultados preliminares en holdout

### Modelo base vs modelo tuneado

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting Base | 0.9625 | 0.9560 | 0.8031 | 0.8729 | 0.9894 |
| Gradient Boosting Tuned | **0.9674** | **0.9544** | **0.8369** | **0.8918** | **0.9918** |

### Matriz de confusión del modelo tuneado

- Verdaderos negativos: **1688**
- Falsos positivos: **13**
- Falsos negativos: **53**
- Verdaderos positivos: **272**

### Lectura de negocio

El modelo tuneado mejora la capacidad de detectar clientes en churn sin incrementar de forma relevante los falsos positivos. Esto lo vuelve una base sólida para una futura aplicación de scoring y retención.

---

## Interpretación del modelo clásico

El análisis de importancia de variables mostró que el modelo final se apoya principalmente en **variables de comportamiento transaccional** y profundidad de relación con el banco.

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

## Estructura del proyecto

```text
credit-card-churn-scoring/
├── notebooks/
├── src/
├── tests/
├── data/
├── models/
├── configs/
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── .env.example
└── README.md

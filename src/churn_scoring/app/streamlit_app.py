"""Aplicación Streamlit para predicción de churn de clientes."""

import streamlit as st

from churn_scoring.app.inference import CustomerInput, predict_customer_churn
from churn_scoring.config import get_settings
from churn_scoring.models.sklearn_model import SklearnChurnModel

EDUCATION_LEVELS = [
    "Graduate",
    "High School",
    "Unknown",
    "Uneducated",
    "College",
    "Post-Graduate",
    "Doctorate",
]

MARITAL_STATUSES = [
    "Married",
    "Single",
    "Divorced",
    "Unknown",
]

INCOME_CATEGORIES = [
    "Less than $40K",
    "$40K - $60K",
    "$60K - $80K",
    "$80K - $120K",
    "$120K +",
    "Unknown",
]

CARD_CATEGORIES = [
    "Blue",
    "Silver",
    "Gold",
    "Platinum",
]


@st.cache_resource
def load_churn_model(model_path: str) -> SklearnChurnModel:
    """Carga el modelo de churn desde un artefacto joblib.

    Parameters
    ----------
    model_path : str
        Ruta del modelo serializado.

    Returns
    -------
    SklearnChurnModel
        Modelo listo para inferencia.
    """
    return SklearnChurnModel(model_path)


def build_sidebar_input() -> CustomerInput:
    """Construye el input del cliente a partir del formulario de Streamlit.

    Returns
    -------
    CustomerInput
        Datos del cliente listos para convertirse a DataFrame.
    """
    st.sidebar.header("Datos del cliente")

    customer_age = st.sidebar.slider("Edad", 18, 100, 45)
    gender = st.sidebar.selectbox("Género", ["M", "F"])
    dependent_count = st.sidebar.slider("Dependientes", 0, 10, 2)

    education_level = st.sidebar.selectbox(
        "Nivel educativo",
        EDUCATION_LEVELS,
    )
    marital_status = st.sidebar.selectbox(
        "Estado civil",
        MARITAL_STATUSES,
    )
    income_category = st.sidebar.selectbox(
        "Categoría de ingreso",
        INCOME_CATEGORIES,
    )
    card_category = st.sidebar.selectbox(
        "Categoría de tarjeta",
        CARD_CATEGORIES,
    )

    st.sidebar.header("Relación con el banco")

    months_on_book = st.sidebar.slider("Meses como cliente", 1, 80, 36)
    total_relationship_count = st.sidebar.slider(
        "Productos contratados",
        1,
        10,
        5,
    )
    months_inactive_12_mon = st.sidebar.slider(
        "Meses inactivo en los últimos 12 meses",
        0,
        12,
        1,
    )
    contacts_count_12_mon = st.sidebar.slider(
        "Contactos en los últimos 12 meses",
        0,
        20,
        2,
    )

    st.sidebar.header("Comportamiento financiero")

    credit_limit = st.sidebar.number_input(
        "Límite de crédito",
        min_value=100.0,
        max_value=100_000.0,
        value=3_000.0,
        step=100.0,
    )
    total_revolving_bal = st.sidebar.number_input(
        "Saldo revolvente",
        min_value=0.0,
        max_value=float(credit_limit),
        value=min(1_000.0, float(credit_limit)),
        step=100.0,
    )
    avg_open_to_buy = credit_limit - total_revolving_bal

    total_amt_chng_q4_q1 = st.sidebar.number_input(
        "Cambio en monto Q4/Q1",
        min_value=0.0,
        max_value=5.0,
        value=1.2,
        step=0.1,
    )
    total_trans_amt = st.sidebar.number_input(
        "Monto total transaccionado",
        min_value=1.0,
        max_value=50_000.0,
        value=2_400.0,
        step=100.0,
    )
    total_trans_ct = st.sidebar.slider(
        "Número total de transacciones",
        1,
        200,
        24,
    )
    total_ct_chng_q4_q1 = st.sidebar.number_input(
        "Cambio en conteo de transacciones Q4/Q1",
        min_value=0.0,
        max_value=5.0,
        value=0.9,
        step=0.1,
    )
    avg_utilization_ratio = st.sidebar.slider(
        "Ratio promedio de utilización",
        0.0,
        1.0,
        0.3,
        step=0.01,
    )

    return CustomerInput(
        customer_age=customer_age,
        gender=gender,
        dependent_count=dependent_count,
        education_level=education_level,
        marital_status=marital_status,
        income_category=income_category,
        card_category=card_category,
        months_on_book=months_on_book,
        total_relationship_count=total_relationship_count,
        months_inactive_12_mon=months_inactive_12_mon,
        contacts_count_12_mon=contacts_count_12_mon,
        credit_limit=credit_limit,
        total_revolving_bal=total_revolving_bal,
        avg_open_to_buy=avg_open_to_buy,
        total_amt_chng_q4_q1=total_amt_chng_q4_q1,
        total_trans_amt=total_trans_amt,
        total_trans_ct=total_trans_ct,
        total_ct_chng_q4_q1=total_ct_chng_q4_q1,
        avg_utilization_ratio=avg_utilization_ratio,
    )


def render_prediction_result(
    churn_probability: float,
    prediction: int,
    risk_level: str,
    recommendation: str,
) -> None:
    """Muestra el resultado de predicción en la aplicación.

    Parameters
    ----------
    churn_probability : float
        Probabilidad estimada de churn.
    prediction : int
        Clase predicha.
    risk_level : str
        Nivel de riesgo calculado.
    recommendation : str
        Recomendación de negocio.
    """
    probability_pct = churn_probability * 100

    col_1, col_2, col_3 = st.columns(3)

    with col_1:
        st.metric("Probabilidad de churn", f"{probability_pct:.2f}%")

    with col_2:
        st.metric("Nivel de riesgo", risk_level)

    with col_3:
        label = "Churn" if prediction == 1 else "Cliente existente"
        st.metric("Clasificación", label)

    st.progress(min(max(churn_probability, 0.0), 1.0))

    st.subheader("Recomendación")
    st.info(recommendation)


def main() -> None:
    """Ejecuta la aplicación principal de Streamlit."""
    st.set_page_config(
        page_title="Credit Card Churn Scoring",
        page_icon="💳",
        layout="wide",
    )

    st.title("Credit Card Churn Scoring")
    st.write(
        """
        Esta aplicación estima la probabilidad de churn de un cliente de tarjeta
        de crédito usando un pipeline de Machine Learning entrenado con
        scikit-learn.
        """
    )

    settings = get_settings()
    model = load_churn_model(str(settings.sklearn_model_path))
    customer_input = build_sidebar_input()

    st.subheader("Cliente capturado")
    st.dataframe(customer_input.to_model_dataframe(), use_container_width=True)

    if st.button("Calcular score de churn", type="primary"):
        prediction = predict_customer_churn(model, customer_input)

        render_prediction_result(
            churn_probability=prediction.churn_probability,
            prediction=prediction.prediction,
            risk_level=prediction.risk_level,
            recommendation=prediction.recommendation,
        )

    st.divider()

    st.subheader("Modelo utilizado")
    st.write(
        """
        Modelo champion actual: **Gradient Boosting Tuned Pipeline**.
        El pipeline incluye preprocesamiento, codificación de variables
        categóricas y modelo final serializado con joblib.
        """
    )

    st.caption(f"Artefacto cargado desde: `{settings.sklearn_model_path}`")


if __name__ == "__main__":
    main()

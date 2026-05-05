"""Aplicación Streamlit para predicción de churn de clientes."""

import joblib
import pandas as pd
import streamlit as st

from churn_scoring.app.inference import CustomerInput, predict_customer_churn
from churn_scoring.config import get_settings
from churn_scoring.models.sklearn_model import SklearnChurnModel
from churn_scoring.models.torch_model import TorchChurnModel

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
    """Carga el modelo champion de churn desde un artefacto joblib.

    Parameters
    ----------
    model_path : str
        Ruta del modelo serializado.

    Returns
    -------
    SklearnChurnModel
        Modelo scikit-learn listo para inferencia.
    """
    return SklearnChurnModel(model_path)


@st.cache_resource
def load_pytorch_churn_model(
    model_path: str,
    preprocessor_path: str,
) -> TorchChurnModel:
    """Carga el modelo PyTorch de churn junto con su preprocesador.

    Parameters
    ----------
    model_path : str
        Ruta del checkpoint ``.pt`` con los pesos del modelo.
    preprocessor_path : str
        Ruta del preprocesador serializado con joblib.

    Returns
    -------
    TorchChurnModel
        Modelo PyTorch listo para inferencia.
    """
    preprocessor = joblib.load(preprocessor_path)
    input_dim = len(preprocessor.get_feature_names_out())

    return TorchChurnModel.from_state_dict(
        model_path,
        input_dim=input_dim,
        feature_columns=[],
        hidden_dims=(64, 32),
        dropout=0.2,
        output_dim=2,
        device="cpu",
        preprocessor=preprocessor,
    )


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
    """Muestra el resultado principal de predicción en la aplicación.

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


def render_model_comparison(
    sklearn_probability: float,
    sklearn_risk_level: str,
    pytorch_probability: float,
    pytorch_risk_level: str,
) -> None:
    """Muestra una comparación experimental entre modelos.

    Parameters
    ----------
    sklearn_probability : float
        Probabilidad de churn estimada por Gradient Boosting.
    sklearn_risk_level : str
        Nivel de riesgo estimado por Gradient Boosting.
    pytorch_probability : float
        Probabilidad de churn estimada por PyTorch MLP.
    pytorch_risk_level : str
        Nivel de riesgo estimado por PyTorch MLP.
    """
    comparison_df = pd.DataFrame(
        {
            "Modelo": ["Gradient Boosting", "PyTorch MLP"],
            "Rol": ["Champion", "Challenger"],
            "Probabilidad de churn": [
                f"{sklearn_probability * 100:.2f}%",
                f"{pytorch_probability * 100:.2f}%",
            ],
            "Nivel de riesgo": [
                sklearn_risk_level,
                pytorch_risk_level,
            ],
        }
    )

    st.dataframe(comparison_df, use_container_width=True)

    chart_df = pd.DataFrame(
        {
            "Modelo": ["Gradient Boosting", "PyTorch MLP"],
            "Probabilidad de churn": [
                sklearn_probability,
                pytorch_probability,
            ],
        }
    ).set_index("Modelo")

    st.bar_chart(chart_df)

    probability_gap = abs(sklearn_probability - pytorch_probability)

    if probability_gap <= 0.10:
        st.success("Ambos modelos muestran una estimación similar de riesgo de churn.")
    else:
        st.warning(
            "Los modelos presentan diferencias relevantes. Esto puede ocurrir "
            "porque las probabilidades no necesariamente están calibradas de la "
            "misma forma entre Gradient Boosting y PyTorch MLP."
        )


def main() -> None:
    """Ejecuta la aplicación principal de Streamlit."""
    st.set_page_config(
        page_title="Credit Card Churn Scoring",
        page_icon="",
        layout="wide",
    )

    st.title("Credit Card Churn Scoring")
    st.write(
        """
        Esta aplicación estima la probabilidad de churn de un cliente de tarjeta
        de crédito. El resultado principal se basa en un modelo
        **Gradient Boosting Tuned Pipeline**, considerado como modelo champion.
        Además, se muestra una comparación experimental contra un modelo
        **PyTorch MLP**.
        """
    )

    settings = get_settings()

    sklearn_model = load_churn_model(str(settings.sklearn_model_path))
    pytorch_model = load_pytorch_churn_model(
        str(settings.pytorch_model_path),
        str(settings.pytorch_preprocessor_path),
    )

    customer_input = build_sidebar_input()

    st.subheader("Cliente capturado")
    st.dataframe(customer_input.to_model_dataframe(), use_container_width=True)

    if st.button("Calcular score de churn", type="primary"):
        sklearn_prediction = predict_customer_churn(
            sklearn_model,
            customer_input,
        )
        pytorch_prediction = predict_customer_churn(
            pytorch_model,
            customer_input,
        )

        st.subheader("Resultado principal: Gradient Boosting Champion")

        render_prediction_result(
            churn_probability=sklearn_prediction.churn_probability,
            prediction=sklearn_prediction.prediction,
            risk_level=sklearn_prediction.risk_level,
            recommendation=sklearn_prediction.recommendation,
        )

        with st.expander("Comparación experimental con PyTorch MLP"):
            st.write(
                """
                El modelo **Gradient Boosting** se considera el modelo champion
                actual porque obtuvo mejor desempeño en las métricas de
                evaluación.

                El modelo **PyTorch MLP** se muestra como challenger para
                comparar un enfoque de red neuronal contra el pipeline clásico.
                Las probabilidades entre modelos pueden diferir porque no
                necesariamente están calibradas de la misma forma.
                """
            )

            render_model_comparison(
                sklearn_probability=sklearn_prediction.churn_probability,
                sklearn_risk_level=sklearn_prediction.risk_level,
                pytorch_probability=pytorch_prediction.churn_probability,
                pytorch_risk_level=pytorch_prediction.risk_level,
            )

    st.divider()

    st.subheader("Modelos utilizados")
    st.write(
        """
        La decisión principal de la aplicación se basa en el modelo
        **Gradient Boosting Tuned Pipeline**, considerado como modelo champion.

        El modelo **PyTorch MLP** se incluye como challenger para mostrar una
        comparación entre un enfoque clásico de machine learning y una red
        neuronal.
        """
    )

    st.caption(f"Modelo scikit-learn: `{settings.sklearn_model_path}`")
    st.caption(f"Modelo PyTorch: `{settings.pytorch_model_path}`")
    st.caption(f"Preprocesador PyTorch: `{settings.pytorch_preprocessor_path}`")


if __name__ == "__main__":
    main()

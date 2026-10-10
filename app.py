import joblib
import pandas as pd
import streamlit as st

model = joblib.load("machine_failure_model.pkl")

import streamlit as st

st.title("Predictive Equipment Monitor")

st.write(
    "Enter the machine's operating measurements "
    "to explore a machine failure prediction."
)

st.subheader("Machine Measurements")



st.subheader("Machine Measurements")

col1, col2 = st.columns(2)

with col1:
    machine_type_label = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        value=300.0
    )

    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        value=1500
    )

with col2:
    process_temp = st.number_input(
        "Process Temperature (K)",
        value=310.0
    )

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        value=100
    )

type_mapping = {"L": 2, "M": 1, "H": 0}
machine_type = type_mapping[machine_type_label]
st.subheader("Machine Failure Prediction")

if st.button("Predict Machine Failure"):

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    failure_probability = probabilities[1]

    st.metric(
        "Estimated Failure Probability",
            f"{failure_probability:.1%}"
    )
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted")
        st.write(
            "The model classified these measurements "
            "as a potential failure. Further inspection may be needed."
        )
    else:
        st.success("✅ No Machine Failure Predicted")
        st.write(
            "The model did not predict a failure for these inputs. "
            "This does not guarantee that the machine is safe."
        )


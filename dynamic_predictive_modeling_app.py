import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    h1 {
        color: #2E86C1;
        text-align: center;
    }
    h2 {
        color: #148F77;
    }
    .stButton button {
        background-color: #2E86C1;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #1B4F72;
    }
    .stDownloadButton button {
        background-color: #148F77;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stDownloadButton button:hover {
        background-color: #0E6251;
    }
    .stSlider {
        color: #2E86C1;
    }
    .stSuccess {
        background-color: #D5F5E3;
        color: #148F77;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: #FADBD8;
        color: #C0392B;
        padding: 10px;
        border-radius: 5px;
    }
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        margin: 1em 0;
        font-size: 1em;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .dataframe th, .dataframe td {
        padding: 12px 15px;
        text-align: center;
        border: 1px solid #000000;
    }
    .dataframe th {
        background-color: #ffffff;
        color: #000000;
        font-weight: bold;
    }
    .dataframe tr {
        background-color: #ffffff;
        color: #000000;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    </style>
    """, unsafe_allow_html=True)

# Generate dynamic sample data
def generate_sample_data():
    np.random.seed(42)
    data = {
        'Age': np.random.randint(18, 65, 100),
        'Income': np.random.normal(50000, 15000, 100),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
        'Purchase': np.random.choice([0, 1], 100),
        'Spending_Score': np.random.randint(1, 100, 100)
    }
    df = pd.DataFrame(data)
    return df

# Preprocess data
def preprocess_data(df):
    # Convert categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    
    # Handle missing values
    df = df.fillna(df.mean())
    return df

# Main app
def main():
    # App title with emoji and custom styling
    st.markdown("<h1>Dynamic Machine Learning App 🤖</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            Upload your CSV file and select target column to build a predictive model.
        </div>
    """, unsafe_allow_html=True)

    # Generate and display sample data
    sample_df = generate_sample_data()
    st.sidebar.markdown("<h2>Sample Data</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("Download the sample CSV to understand the required format.")
    st.sidebar.download_button(
        label="📥 Download Sample CSV",
        data=sample_df.to_csv(index=False).encode(),
        file_name='dynamic_sample.csv',
        mime='text/csv',
        key='download-sample'
    )

    # File upload section
    st.markdown("<h2>Upload Your Data</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file-uploader")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("<h2>Data Preview</h2>", unsafe_allow_html=True)
        
        # Display the table in black and white
        st.dataframe(df.head().style
            .set_properties(**{'background-color': '#ffffff', 'color': '#000000', 'border': '1px solid #000000'})
        )

        # Preprocess data
        df_processed = preprocess_data(df.copy())

        # Select target column
        st.markdown("<h2>Model Configuration</h2>", unsafe_allow_html=True)
        target_col = st.selectbox("Select Target Column", df.columns, key="target-select")

        if target_col:
            try:
                # Split features and target
                X = df_processed.drop(columns=[target_col])
                y = df_processed[target_col]

                # Split data
                test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, key="test-size-slider")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # Train model
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                st.markdown("<h2>Model Performance</h2>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                with col2:
                    st.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")

                # Prediction section
                st.markdown("<h2>Make Predictions</h2>", unsafe_allow_html=True)
                input_data = {}
                for col in X.columns:
                    if df[col].dtype == 'object':
                        input_data[col] = st.selectbox(col, df[col].unique(), key=f"input-{col}")
                    else:
                        input_data[col] = st.number_input(col, value=df[col].mean(), key=f"input-{col}")

                if st.button("🚀 Predict", key="predict-button"):
                    input_df = pd.DataFrame([input_data])
                    input_processed = preprocess_data(input_df)
                    prediction = model.predict(input_processed)
                    st.markdown(f"""
                        <div class="stSuccess">
                            Predicted <strong>{target_col}</strong>: <strong>{prediction[0]:.2f}</strong>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                    <div class="stError">
                        Error: {str(e)}
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

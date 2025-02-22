# Dynamic ML Prediction App 🚀

A Streamlit-based web application for dynamic machine learning modeling with CSV data upload capabilities. The app allows users to upload their dataset, preprocess it, train a Random Forest Regressor, and make predictions in real-time.

Screenshot of the model:

![image](https://github.com/user-attachments/assets/f6b9ccfc-64f1-456a-98aa-18e529300ab1)
![image](https://github.com/user-attachments/assets/4342eaa4-5a54-4c56-887d-8693b0ebfab0)


## Features ✨
- **CSV File Upload**: Upload your dataset in CSV format.
- **Automatic Preprocessing**: Handles categorical encoding and missing values.
- **Dynamic Target Selection**: Choose the target column for prediction.
- **Random Forest Regression**: Train a model with customizable test size.
- **Real-Time Predictions**: Input feature values and get predictions instantly.
- **Performance Metrics**: View R² Score and Mean Squared Error.
- **Sample Data**: Download a sample CSV to understand the required format.
- **Clean Black & White Table**: Minimalistic and professional data preview.

Flow of the Machine Learning App 🚀
📂 1. User Uploads CSV (Streamlit, Pandas)

Supports up to 200MB
Displays data preview using st.dataframe()
⚙️ 2. Data Preprocessing (Pandas, NumPy, Scikit-learn)

Handles missing values (fill with mean)
Encodes categorical columns (LabelEncoder)
🛠 3. Model Configuration (Scikit-learn, Streamlit Widgets)

User selects target column (st.selectbox())
Chooses train-test split ratio (st.slider())
🤖 4. Model Training (Scikit-learn, RandomForestRegressor)

Splits data into training & testing sets (train_test_split())
Trains Random Forest model (RandomForestRegressor())
📊 5. Model Evaluation (Scikit-learn, Streamlit Metrics)

Calculates R² Score & Mean Squared Error (r2_score(), mean_squared_error())
Displays performance metrics using st.metric()
📡 6. Make Predictions (Scikit-learn, Pandas, Streamlit Inputs)

User inputs new data (st.number_input(), st.selectbox())
Model predicts target value (model.predict())
Displays prediction in styled output (st.markdown())
🎨 7. User Interface & Styling (Streamlit, HTML, CSS)

Custom button styles, colors, layout using st.markdown()
Table styling for data display
📥 8. Sample Data Download (Pandas, Streamlit Download Button)

User can download example CSV (st.download_button())

## Requirements 📦
- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy

## Installation ⚙️
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo

##Install the required dependencies:

bash
Copy
pip install -r requirements.txt   

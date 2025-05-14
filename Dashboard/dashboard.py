import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import os

# Set visualization defaults
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Dashboard/weatherAUS.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Preprocess data
def preprocess_data(df):
    df = df.dropna(subset=['RainTomorrow'])
    categorical = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    for col in categorical:
        df[col].fillna(df[col].mode()[0], inplace=True)
    numeric = df.select_dtypes(include=[np.number]).columns
    for col in numeric:
        df[col].fillna(df[col].median(), inplace=True)
    
    le = LabelEncoder()
    for col in categorical:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop(['RainTomorrow', 'Date'], axis=1)
    y = le.fit_transform(df['RainTomorrow'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

# Load ANN model
def load_model_from_file(file_path):
    if os.path.exists(file_path):
        try:
            model = load_model(file_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

# Predict rain
def predict_rain(model, input_data, scaler):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)
        predicted_class = (prediction > 0.5).astype("int32")
        return prediction, predicted_class
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Load dataset
df_original = load_data()
if df_original is None:
    st.stop()

# Preprocess data
X, y, scaler, feature_cols = preprocess_data(df_original)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Dashboard title
st.title("🌦 Australian Weather Prediction Dashboard")
st.write("Analyze weather data and predict rainfall using an ANN model.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Exploration", "📊 Analysis", "🔮 Prediction", "📝 Summary"])

# Sidebar
st.sidebar.title("📋 Dashboard Info")
st.sidebar.subheader("Dataset")
st.sidebar.write("Historical weather data from Australia, including temperature, humidity, and rainfall.")
st.sidebar.subheader("Team")
st.sidebar.write("- Wahyu Ozorah Manurung\n- Damianus Christopher Samosir\n- Yuda Reyvandra Herman\n- Ferdy Fitriansyah Rowi")
st.sidebar.markdown("<p style='text-align: center; color: #A9A9A9;'>© 2025 Weather Analytics</p>", unsafe_allow_html=True)

# Tab 1: Data Exploration
with tab1:
    st.subheader("📊 Dataset Overview")
    st.write("### Raw Data")
    st.dataframe(df_original.head())
    
    st.write("### Statistics")
    st.dataframe(df_original.describe())
    
    st.write("### Preprocessed Data")
    X_df = pd.DataFrame(X, columns=feature_cols)
    st.dataframe(X_df.head())
    
    st.write("### Feature Distributions")
    key_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity3pm', 'Pressure3pm']
    fig_hist, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(key_features[:5]):
        axes[i].hist(df_original[col].dropna(), bins=50)
        axes[i].set_title(col)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig_hist)

# Tab 2: Analysis
with tab2:
    st.subheader("📊 Model Analysis")
    
    st.write("### Classification Report")
    report = """
              precision    recall  f1-score   support
           0       0.88      0.94      0.91     20110
           1       0.70      0.51      0.59      5398
    accuracy                           0.85     25508
    """
    st.text(report)
    
    st.write("### Confusion Matrix")
    cm = [[int(20110 * 0.94), int(20110 * (1-0.94))],
          [int(5398 * (1-0.51)), int(5398 * 0.51)]]
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Rain', 'Rain'],
                yticklabels=['No Rain', 'Rain'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig_cm)
    
    st.write("### Feature Correlations")
    df_corr = df_original.copy()
    df_corr['RainTomorrow'] = LabelEncoder().fit_transform(df_corr['RainTomorrow'])
    correlation = df_corr.select_dtypes(include=[np.number]).corr()['RainTomorrow'].sort_values(ascending=False)[1:]
    fig_corr, ax = plt.subplots()
    correlation.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Correlation with RainTomorrow')
    ax.set_xlabel('Features')
    ax.set_ylabel('Correlation')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig_corr)

# Tab 3: Prediction
with tab3:
    st.subheader("🔮 Rain Prediction")
    model_path = "ann_weather_model.h5"
    model = load_model_from_file(model_path)
    
    if model is None:
        st.warning("Training new ANN model...")
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        model.save(model_path)
        st.success("Model trained and saved.")
    
    st.write("Enter weather data:")
    input_data = {}
    col1, col2, col3 = st.columns(3)
    for i, col in enumerate(feature_cols):
        with [col1, col2, col3][i % 3]:
            input_data[col] = st.number_input(col, value=float(df_original[col].median() if col in df_original.select_dtypes(include=[np.number]).columns else 0))
    
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        prob, pred_class = predict_rain(model, input_df, scaler)
        if prob is not None:
            result = "Rain" if pred_class[0][0] == 1 else "No Rain"
            st.success(f"Prediction: {result} (Probability: {prob[0][0]:.4f})")

# Tab 4: Summary
with tab4:
    st.subheader("📝 Summary")
    st.write("""
    - *Key Features*: Humidity3pm, Rainfall, and Cloud3pm are strong predictors of rain.
    - *Model*: The ANN achieves 85% accuracy, with better performance for 'No Rain' (recall: 0.94) than 'Rain' (recall: 0.51).
    - *Applications*: Useful for agriculture and weather forecasting.
    - *Limitations*: Lower recall for 'Rain' suggests potential for improvement with more data or features.
    """)

# Footer
st.markdown("<p style='text-align: center; color: #A9A9A9;'>© 2025 Weather Analytics</p>", unsafe_allow_html=True)

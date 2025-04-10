import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="🎯 Student Performance Analyzer", layout="wide")
st.title("🎓 Student Performance Prediction & Analysis")
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #4B8BBE; }
    .css-1d391kg { background-color: white; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Preview of Dataset")
    st.dataframe(df.head())

    # Explanation of abbreviations
    with st.expander("🔍 Abbreviations & Feature Descriptions"):
        st.markdown("""
        **Personal Info:**
        - `school`: Student's school (GP = Gabriel Pereira, MS = Mousinho da Silveira)
        - `sex`: Gender (F = Female, M = Male)
        - `age`: Age in years
        - `address`: Home address (U = Urban, R = Rural)
        - `famsize`: Family size (LE3 = ≤3 members, GT3 = >3 members)
        - `Pstatus`: Parent's cohabitation status (T = Together, A = Apart)

        **Parental Background:**
        - `Medu`: Mother's education (0 = none to 4 = higher)
        - `Fedu`: Father's education
        - `Mjob` / `Fjob`: Mother's/Father's job (teacher, health, services, etc.)
        - `reason`: Reason for choosing this school
        - `guardian`: Guardian (mother, father, other)

        **Academic Behavior:**
        - `studytime`: Weekly study time (1 = <2h to 4 = >10h)
        - `failures`: Past class failures
        - `schoolsup` / `famsup`: School/family support (yes/no)
        - `paid`: Extra paid classes
        - `activities`: Extracurricular activities
        - `higher`: Intends higher education
        - `internet`: Internet access at home
        - `romantic`: In a romantic relationship

        **Lifestyle:**
        - `famrel`: Family relationship quality (1-5)
        - `freetime`: Free time after school (1-5)
        - `goout`: Going out frequency (1-5)
        - `Dalc` / `Walc`: Daily/Weekend alcohol use (1-5)
        - `health`: Health status (1-5)
        - `absences`: School absences

        **Grades:**
        - `G1`, `G2`: First and second period grades (0-20)
        - `G3`: Final grade (Target for prediction)

        **Metrics:**
        - `Accuracy`: Overall correctness of prediction
        - `Precision`: Correctness of positive predictions
        - `Recall`: Ability to find all positive cases
        - `F1 Score`: Balance between precision and recall
        """)

    if 'G3' not in df.columns:
        st.error("❌ 'G3' column (final grade) not found in dataset!")
    else:
        # Encode categorical variables
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

        # Descriptive statistics
        st.subheader("📊 Descriptive Statistics")
        st.dataframe(df_encoded.describe())

        # Correlation Heatmap
        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_encoded.corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Regression
        st.subheader("📈 Regression Analysis (Predicting G3)")
        X = df_encoded.drop('G3', axis=1)
        y = df_encoded['G3']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("📉 Mean Absolute Error", f"{mae:.2f}")
        col2.metric("📏 RMSE", f"{rmse:.2f}")
        col3.metric("📊 R² Score", f"{r2:.2f}")

        st.subheader("🧮 Grade Prediction Results")
        results_df = X_test.copy()
        results_df['Actual G3'] = y_test
        results_df['Predicted G3'] = y_pred.round(2)
        st.dataframe(results_df)

        # Classification: Pass/Fail
        st.subheader("✅ Classification: Predicting Pass/Fail")
        df_encoded['pass'] = df_encoded['G3'].apply(lambda x: 1 if x >= 10 else 0)
        X_cls = df_encoded.drop(['G3', 'pass'], axis=1)
        y_cls = df_encoded['pass']
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_cls, y_train_cls)
        cls_preds = clf.predict(X_test_cls)

        acc = accuracy_score(y_test_cls, cls_preds)
        prec = precision_score(y_test_cls, cls_preds)
        rec = recall_score(y_test_cls, cls_preds)
        f1 = f1_score(y_test_cls, cls_preds)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Accuracy", f"{acc:.2%}")
        col2.metric("✅ Precision", f"{prec:.2%}")
        col3.metric("🔍 Recall", f"{rec:.2%}")
        col4.metric("📎 F1 Score", f"{f1:.2%}")

        st.text("📋 Detailed Classification Report")
        st.code(classification_report(y_test_cls, cls_preds))

        # Downloadable Dropout List
        st.subheader("📤 Download Student Dropout List")
        dropouts = df[df['G3'] < 10]
        st.write("Students considered dropouts (G3 < 10):", dropouts.shape[0])
        st.dataframe(dropouts)

        csv = dropouts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dropout List as CSV",
            data=csv,
            file_name='dropout_students.csv',
            mime='text/csv'
        )

# 🎓 Student Performance Analyzer

An interactive Streamlit application that allows users to upload student academic datasets, explore patterns, visualize insights, and predict student performance (grades and pass/fail status) using machine learning models.

---

## 🚀 Features

- 📁 **CSV Upload**: Upload your own dataset with student performance metrics.
- 🧠 **Automatic Encoding**: Categorical features are automatically label encoded.
- 📊 **Descriptive Statistics**: Explore numerical summaries of all features.
- 🔥 **Correlation Heatmap**: Visualize relationships between variables.
- 📈 **Regression Prediction (G3 - Final Grade)**:
  - Predict student final grades using **Linear Regression**.
  - Evaluation metrics: MAE, RMSE, R² Score.
- ✅ **Classification (Pass/Fail)**:
  - Predict whether a student will pass or fail using **Logistic Regression**.
  - Evaluation metrics: Accuracy, Precision, Recall, F1 Score.
- 📤 **Dropout Detection**:
  - Automatically detects students with G3 < 10 as potential dropouts.
  - Option to **download dropout list** as a CSV.

---

## 📦 Requirements

To run the application, you’ll need:

```bash
pip install streamlit pandas seaborn matplotlib scikit-learn numpy
```

---

## 🧠 Model Information

- **Linear Regression**: Predicts numeric final grade (`G3`).
- **Logistic Regression**: Classifies pass (`G3` ≥ 10) or fail (`G3` < 10).

---

## 📁 Dataset Format

Your CSV should include the following key columns (among others):

- Personal Info: `school`, `sex`, `age`, `address`, etc.
- Parental Background: `Medu`, `Fedu`, `Mjob`, `Fjob`, etc.
- Academic Behavior: `studytime`, `failures`, `schoolsup`, `higher`, etc.
- Lifestyle: `goout`, `Dalc`, `Walc`, `health`, `absences`
- Grades: `G1`, `G2`, `G3` *(final grade is required for predictions)*

📌 **Note**: The column `G3` is mandatory for prediction and evaluation.

---

## 💡 How to Run

1. Save the script as `app.py`.
2. Run the app with Streamlit:

```bash
streamlit run app.py
```

3. Upload your dataset in CSV format when prompted.

---


## 📥 Download

Get the predicted dropout list in a downloadable `.csv` format from the app interface.

---

## 🧑‍💻 Author

**Sri Prabhath Reddy Bhuma**

---

## 📄 License

This project is open source and free to use for educational and research purposes.

---

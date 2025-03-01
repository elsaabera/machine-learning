

# 🎓 Student Dropout Prediction - Machine Learning Project  

Student dropout is a major issue in the education sector, affecting both institutions and students. Early identification of at-risk students can help implement timely interventions. This project leverages Machine Learning to predict student dropout rates based on academic performance, socio-economic background, and behavioral factors.  

🔗 Live API Documentation: [Click Here](https://machine-learning-1gt1.onrender.com/docs)  

---

## 🚀 Problem Definition  
The goal is to develop a predictive model that identifies students at risk of dropping out using various academic, socio-economic, and behavioral indicators. By leveraging machine learning, institutions can take proactive steps to reduce dropout rates and improve student retention.  

---

## 🎯 Objectives  
✔️ Develop a predictive model to classify students based on their likelihood of dropping out.  
✔️ Perform Exploratory Data Analysis (EDA) to gain insights into dropout patterns.  
✔️ Preprocess data by handling missing values, encoding categorical variables, and normalizing numerical features.  
✔️ Train and compare multiple Machine Learning models to find the best-performing algorithm.  
✔️ Optimize the model using Hyperparameter Tuning and Cross-Validation.  
✔️ Deploy the trained model as an API using FastAPI for real-time predictions.  

---

## 📊 Data Description  
📌 Source: Educational dataset containing student demographics, academic performance, financial background, and engagement levels.  

📌 Structure: CSV format with key features:  
- 🧑‍🎓 Demographics: Age, gender, nationality, parental education.  
- 📚 Academic Performance: GPA, exam scores, course completion rates.  
- 📅 Behavioral Data: Attendance records, engagement levels, disciplinary records.  
- 💰 Socio-Economic Factors: Scholarship status, employment status.  
- 🎯 Target Variable: Dropout status (1 = Dropped out, 0 = Continued).  

---

## 📈 Exploratory Data Analysis (EDA)  
EDA is conducted to understand data distributions, detect missing values, and identify key trends.  

🔹 Visualizations: Histograms, box plots, and correlation matrices.  
🔹 Missing Values Handling: Imputation strategies applied where necessary.  
🔹 Feature Correlations: Analyzing relationships between features and dropout rates.  
🔹 Class Imbalance Detection: Identifying whether dropout cases are underrepresented.  

---

## 🛠 Preprocessing Steps  
✔️ Handling missing values via imputation.  
✔️ Standardizing numerical variables.  
✔️ Encoding categorical features using One-Hot Encoding or Label Encoding.  
✔️ Splitting dataset into 80% training and 20% testing sets.  
✔️ Addressing class imbalances using SMOTE or class weighting.  

---

## 🤖 Model Implementation  
Several machine learning models are trained and compared:  
✅ Logistic Regression  
✅ Decision Trees  
✅ Random Forest  
✅ Support Vector Machines (SVM)  
✅ Neural Networks  

---

## ⚙️ Hyperparameter Tuning  
🔹 Grid Search & Random Search: Used for optimizing hyperparameters.  
🔹 Cross-Validation: Applied to improve model generalization.  

---

## 📊 Evaluation Metrics  
🔹 Performance Metrics:  
   - Accuracy  
   - Precision, Recall, F1-score  
   - AUC-ROC curve for classification effectiveness  

🔹 Visualization Techniques:  
   - Confusion Matrix  
   - ROC and Precision-Recall Curves  
   - Feature Importance Analysis  

🔹 Baseline Comparison:  
   - Performance evaluated against a simple rule-based classifier.  

---


```json
{
  "data": [
    {
      "Marital_status": 1,
      "Application_mode": 1,
      "Application_order": 1,
      "Course": 1,
      "Daytime_evening_attendance": 1,
      "Previous_qualification": 1,
      "Nacionality": 1,
      "Mothers_qualification": 1,
      "Fathers_qualification": 1,
      "Mothers_occupation": 1,
      "Fathers_occupation": 1,
      "Displaced": 0,
      "Educational_special_needs": 0,
      "Debtor": 0,
      "Tuition_fees_up_to_date": 1,
      "Gender": 1,
      "Scholarship_holder": 0,
      "Age_at_enrollment": 18,
      "International": 0,
      "Curricular_units_1st_sem_credited": 5,
      "Curricular_units_1st_sem_enrolled": 6,
      "Curricular_units_1st_sem_evaluations": 6,
      "Curricular_units_1st_sem_approved": 5,
      "Curricular_units_1st_sem_grade": 12.5,
      "Curricular_units_1st_sem_without_evaluations": 0,
      "Curricular_units_2nd_sem_credited": 5,
      "Curricular_units_2nd_sem_enrolled": 6,
      "Curricular_units_2nd_sem_evaluations": 6,
      "Curricular_units_2nd_sem_approved": 5,
      "Curricular_units_2nd_sem_grade": 11.0,
      "Curricular_units_2nd_sem_without_evaluations": 0,
      "Unemployment_rate": 5.2,
      "Inflation_rate": 1.8,
      "GDP": 3.0
    }]}
```
```

## 🚀 Deployment  
The model is deployed as an API using FastAPI to enable real-time predictions.  

### 🌍 API Functionality  
✅ Accepts student data as input.  
✅ Returns a probability score indicating dropout risk.  
✅ Allows easy integration with education management systems.

### 🔧 Deployment Steps  
1️⃣ Start the API Server:  
uvicorn elsa:app --host 0.0.0.0 --port 8000 --reload


```md
## 🏗 How to Run Locally  

1️⃣ **Clone the repository:**  
```sh
git clone https://github.com/elsaabera/machine-learning.git
cd machine-learning
```

2️⃣ **Create and activate a virtual environment:**  
```sh
python -m venv env
source env/bin/activate  # On Windows use env\Scripts\activate
```

3️⃣ **Install dependencies:**  
```sh
pip install -r requirements.txt
```

4️⃣ **Train the model:**  
```sh
python train.py
```

5️⃣ **Start the API:**  
```sh
uvicorn elsa:app --reload
```

6️⃣ **API Documentation:**  
[Machine Learning API](https://machine-learning-1gt1.onrender.com/docs)
```

## 📦 Dependencies  
- Python 3.x  
- pandas, numpy, scikit-learn  
- fastapi, uvicorn, joblib, matplotlib  

---

## ⚠️ Limitations & Future Improvements  
📌 Data Quality: The dataset may have biases or missing values affecting model accuracy.  
📌 Generalizability: The model might not generalize well to different educational institutions.  
📌 Feature Expansion: Incorporating additional data sources, such as real-time student interactions.  
📌 Continuous Learning: Implementing adaptive models that evolve with new data.  

---

## 📜 License  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

🔗 Live API Documentation: [Machine Learning API](https://machine-learning-1gt1.onrender.com/docs)  

---

 👩‍💻 Author  

Elsa Abera  



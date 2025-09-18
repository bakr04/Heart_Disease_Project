# 🩺 Heart Disease Prediction Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License: MIT">
</p>

> A comprehensive machine learning project that predicts the probability of heart disease from clinical data. This serves as a capstone project for the AI and Machine Learning program at Sprints, demonstrating an end-to-end ML workflow.

---

### ► Overview

This repository walks through the entire machine learning pipeline, including:

* **Data Preprocessing:** Cleaning and preparing the dataset for modeling.
* **Feature Engineering:** Selecting the most impactful features for prediction.
* **Model Development:** Building and training various supervised and unsupervised models.
* **Hyperparameter Tuning:** Optimizing models for the best performance.
* **Deployment:** Serving the best model via a simple Streamlit web application.

---

### ► Project Structure

The repository is organized as follows:

/
├── 📂 data/
│   ├── 📄 final_processing_afterfeatureselection.csv
│   ├── 📄 heart_disease.csv
│   ├── 📄 pca-processed_heart_disease.csv
│   └── 📄 preprocessed_heart_disease.csv
├── 📂 deployment/
│   └── 📄 ngrok_setup.txt
├── 📂 models/
│   ├── 📦 Decision_Tree_optimized.pkl
│   ├── 📦 LinearSVC_optimized.pkl
│   ├── 📦 Logistic_Regression_optimized.pkl
│   ├── 📦 Random_Forest_optimized.pkl
│   ├── 📦 SVM_optimized.pkl
│   ├── 📦 minmax_scaler.pkl
│   ├── 📦 onehot_encoder.pkl
│   ├── 📦 pca.pkl
│   ├── 📦 supervised_decisiontree.pkl
│   ├── 📦 supervised_linearsvc.pkl
│   ├── 📦 supervised_logisticregression.pkl
│   ├── 📦 supervised_randomforest.pkl
│   ├── 📦 supervised_svc.pkl
│   ├── 📦 unsupervised_hierarchicalclustering.pkl
│   └── 📦 unsupervised_kmeans.pkl
├── 📂 notebooks/
│   ├── 📜 01_data_preprocessing.ipynb
│   ├── 📜 02_pca_analysis.ipynb
│   ├── 📜 03_feature_selection.ipynb
│   ├── 📜 04_supervised_learning.ipynb
│   ├── 📜 05_unsupervised_learning.ipynb
│   └── 📜 06_hyperparameter_tuning.ipynb
├── 📂 results/
│   └── 📊 evaluation_metrics.txt
├── 📂 ui/
│   └── 🚀 app.py
├── 📄 .gitignore
├── 📄 LICENSE
├── 📄 README.md
└── 📄 requirements.txt

---

### ► Tech Stack

This project leverages a variety of open-source technologies:

* **Core:** Python 3.13.7+
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Deployment:** Ngrok
* **Development:** Jupyter Notebook

---

### ► Getting Started

To get the project up and running locally, follow these simple steps:

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/nesringamal/Heart_Disease_Project.git](https://github.com/nesringamal/Heart_Disease_Project.git)
    cd Heart_Disease_Project
    ```

2.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Launch the Streamlit application:**
    ```sh
    cd ui
    streamlit run app.py
    ```

4.  **Deploy with Ngrok:**
    For instructions on how to deploy the application publicly, see `deployment/ngrok_setup.txt`.

---

### ► Model Performance

Multiple classification models were trained and evaluated. The **Random Forest Classifier** was selected for the final application due to its superior performance.

==========================================
|| Supervised Models Evaluation Metrics ||
==========================================

----------Logistic Regression----------------------------------
Accuracy: 0.8852459016393442
Precision: 0.8387096774193549
Recall: 0.9285714285714286
f1 Score: 0.8813559322033898
AUC: 0.9577922077922079

✅ Logistic Regression optimized
   CV Best F1 Score: 0.8422
   Test F1 Score: 0.9017
   Best Params: {'solver': 'saga', 'penalty': 'l2', 'max_iter': 5000, 'fit_intercept': False, 'C': 10}
---------------------------------------------------------------

-------Decision Tree Classifier--------------------------------
Precision: 0.84375
Recall: 0.9642857142857143
f1 Score: 0.9
AUC: 0.9063852813852814

✅ Decision Tree optimized
   CV Best F1 Score: 0.7915
   Test F1 Score: 0.8527
   Best Params: {'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 10, 'criterion': 'entropy'}
---------------------------------------------------------------

--------Random Forest Classifier------- (Used Model For GUI)---
Accuracy: 0.9180327868852459
Precision: 0.8709677419354839
Recall: 0.9642857142857143
f1 Score: 0.9152542372881356
AUC: 0.9404761904761905

✅ Random Forest optimized
   CV Best F1 Score: 0.8091
   Test F1 Score: 0.9345
   Best Params: {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': None}
---------------------------------------------------------------

---------------Linear SVC--------------------------------------
Accuracy: 0.9016393442622951
Precision: 0.84375
Recall: 0.9642857142857143
f1 Score: 0.9
AUC: 0.9599567099567099

✅ LinearSVC optimized
   CV Best F1 Score: 0.8337
   Test F1 Score: 0.8854
   Best Params: {'multi_class': 'ovr', 'max_iter': 2000, 'loss': 'squared_hinge', 'C': 0.1}
---------------------------------------------------------------

------------------SVC------------------------------------------
Accuracy: 0.8688524590163934
Precision: 0.8571428571428571
Recall: 0.8571428571428571
f1 Score: 0.8571428571428571
AUC: 0.9242424242424242

✅ SVC optimized
   CV Best F1 Score: 0.8383
   Test F1 Score: 0.8362
   Best Params: {'kernel': 'rbf', 'gamma': 'scale', 'C': 10}
---------------------------------------------------------------


============================================
|| Unsupervised Models Evaluation Metrics ||
============================================

----------------KMeans Clustering-------------------------------
K=2
The average silhouette_score is: 0.4319578463726729
Adjusted Rand Index (ARI): 0.4294611176020748
Normalized Mutual Info (NMI): 0.33850070289267453
Clustering Accuracy Comparing to Labels: 0.8283828382838284

K = 15
The average silhouette_score is: 0.5117095286830087
Adjusted Rand Index (ARI): 0.06250964552364835
Normalized Mutual Info (NMI): 0.18197388233695158
----------------------------------------------------------------


----------------Hierarchical Clustering-------------------------
Hierarchical Clustering vs Target:
Cluster/ Target     0    1          
      0            122   18
      1             42  121 

Adjusted Rand Index (ARI): 0.363
Adjusted Mutual Info (AMI): 0.296
Silhouette Score: 0.424
----------------------------------------------------------------

---

### ► Future Work

Potential areas for future development include:

* **Advanced Modeling:** Experiment with deep learning architectures (e.g., neural networks).
* **Scalability:** Train the models on a larger, more diverse dataset.
* **Cloud Deployment:** Host the application on a cloud platform like AWS or Heroku for 24/7 availability.

---

### ► Author

**Mostafa Bakr**

# 🩺 Heart Disease Prediction Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13.7%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache-yellow?style=for-the-badge" alt="License: Apache License">
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
    gh repo clone bakr04/Heart_Disease_Project
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

#### Supervised Models

| Model                 | Accuracy | Precision | Recall  | F1 Score | AUC     | Optimized Test F1 |
| :-------------------- | :------: | :-------: | :-----: | :------: | :-----: | :---------------: |
| **Random Forest** | **91.8%**| **87.1%** | **96.4%** | **91.5%**| **94.0%** | **93.5%** |
| Logistic Regression   | 88.5%    | 83.9%     | 92.9%   | 88.1%    | 95.8%   | 90.2%             |
| Decision Tree         | -        | 84.4%     | 96.4%   | 90.0%    | 90.6%   | 85.3%             |
| Linear SVC            | 90.2%    | 84.4%     | 96.4%   | 90.0%    | 96.0%   | 88.5%             |
| SVC                   | 86.9%    | 85.7%     | 85.7%   | 85.7%    | 92.4%   | 83.6%             |

<br>

#### Unsupervised Models

| Model            | Metric                 | Value   |
| :--------------- | :--------------------- | :-----: |
| **KMeans (K=2)** | Silhouette Score       | 0.432   |
|                  | Adjusted Rand Index    | 0.429   |
|                  | Normalized Mutual Info | 0.339   |
|                  | Clustering Accuracy    | 82.8%   |
|                                                     |
| **KMeans (K=15)** | Silhouette Score      | 0.512   |
|                  | Adjusted Rand Index    | 0.063   |
|                  | Normalized Mutual Info | 0.182   |
|                                                     |
| **Hierarchical** | Adjusted Rand Index    | 0.363   |
|                  | Adjusted Mutual Info   | 0.296   |
|                  | Silhouette Score       | 0.424   |

---

---

### ► Future Work

Potential areas for future development include:

* **Advanced Modeling:** Experiment with deep learning architectures (e.g., neural networks).
* **Scalability:** Train the models on a larger, more diverse dataset.

---

### ► Author

**Mostafa Bakr**

Linkedin: { www.linkedin.com/in/bakr04 }

Github: { https://github.com/bakr04 }

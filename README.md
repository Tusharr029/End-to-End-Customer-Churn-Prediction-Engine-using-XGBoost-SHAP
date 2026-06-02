# End-to-End Customer Churn Prediction Engine

An enterprise-grade, data-driven machine learning pipeline that identifies high-risk e-commerce customers. The system processes transactional histories, aggregates behavioral consumer data via relational time-bounded queries to eliminate data leakage, constructs an optimized gradient-boosted tree framework with cross-validation tuning, and interprets user risks utilizing modern Explainable AI.

---

## 📊 Pipeline Architecture Overview
The project is built around an end-to-end operational data loop spanning three core operational layers:
1. **Data Engineering Layer (MySQL):** Ingests raw transactional invoice logs and performs time-split RFM (Recency, Frequency, Monetary) matrix aggregations to prevent target data leakage.
2. **Predictive Analytics Layer (XGBoost):** Classifies complex consumer behaviors using hyperparameter-optimized estimators tuned via Stratified 5-Fold Cross-Validation.
3. **Algorithability Transparency Layer (SHAP):** Leverages Explainable AI (XAI) to extract individual root-cause risk drivers for executive decision-making.

---

## 🛠️ Technology Stack & Dependencies
* **Database Management:** MySQL Server & MySQL Workbench
* **Language Environment:** Python 3.12+
* **Core Machine Learning Engines:** `xgboost`, `scikit-learn`
* **Data Engineering & Explainability Frameworks:** `pandas`, `shap`, `sqlalchemy`, `pymysql`

---

## 🏗️ Step-by-Step Implementation Detail

### 1. Database Schema Ingestion & Feature Engineering
A dataset consisting of 50,000 transaction records was mapped into an indexed relational table schema inside MySQL Workbench. 

To ensure complete statistical integrity and avoid data leakage, features are calculated within a strict historical observation window, while targets are labeled based on independent subsequent transaction windows:
* **Recency:** Computes the total number of days elapsed since the user's most recent transaction up to the observation cutoff.
* **Frequency:** Dynamically counts the total volume of successful invoices generated per customer account during the active window.
* **Monetary:** Aggregates and rounds the total lifetime monetary expenditure across active transaction histories.
* **Target Binary Mapping (Churned):** Maps a separate prediction window to mark customers inactive (1) if they generated zero subsequent transaction instances.

### 2. Predictive AI & Class Imbalance Mitigation
The dataset is processed using Python via an **XGBoost Classifier** ensemble. Class imbalance is handled natively via dynamic positional matrix scaling (`scale_pos_weight`) to optimize penalization bounds for false negatives. 

* **Hyperparameter Optimization:** Rather than relying on arbitrary parameters, the system executes an automated `GridSearchCV` tuning grid over tree depth, estimators, and learning rates to secure the highest out-of-sample F1-score.

---

## 📈 System Execution & Performance Results

### Key Performance Indicators (KPIs)
* **Risk Engine Framework:** Hyperparameter-Tuned XGBoost Tree Classifier
* **Validation Rigor:** 5-Fold Cross-Validation Stratified Split
* **Evaluation Metrics Tracked:** Confusion Matrix, Precision, Recall, F1-Score, and SHAP Additive Game-Theory Values

---

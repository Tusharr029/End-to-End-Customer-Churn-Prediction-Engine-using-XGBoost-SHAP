# End-to-End-Customer-Churn-Prediction-Engine-using-XGBoost-SHAP

An enterprise-grade, data-driven machine learning pipeline that identifies high-risk e-commerce customers. The system processes chaotic transactional histories, aggregates behavioral consumer data via relational queries, constructs a predictive gradient-boosted tree framework, and interprets user risks utilizing modern Explainable AI.

## 📊 Pipeline Architecture Overview
The project is built around an end-to-end operational data loop spanning three core operational layers:
1. **Data Engineering Layer (MySQL):** Ingests over 50,000 raw transactional invoice logs and performs RFM (Recency, Frequency, Monetary) matrix aggregations.
2. **Predictive Analytics Layer (XGBoost):** Classifies complex, non-linear consumer behaviors to isolate retention threats.
3. **Algorithmic Transparency Layer (SHAP):** Leverages Explainable AI (XAI) to extract individual root-cause risk drivers for executive decision-making.

---

## 🛠️ Technology Stack & Dependencies
* **Database Management:** MySQL Server & MySQL Workbench
* **Language environment:** Python 3.12+
* **Core Machine Learning Engines:** `xgboost`, `scikit-learn`
* **Data Engineering & Explainability Frameworks:** `pandas`, `shap`, `sqlalchemy`, `pymysql`

---

## 🏗️ Step-by-Step Implementation Detail

### 1. Database Schema Ingestion & Feature Engineering
A dataset consisting of **50,000 transaction records** was mapped into an indexed relational table schema inside MySQL Workbench. 

To convert scattered transactional invoices into centralized profiles, an optimized structural SQL script compiles individual user activities into static **RFM behavioral metrics**:

```sql
CREATE TABLE ai_churn_features AS
SELECT 
    User_Name,
    MAX(Age) AS Age,
    DATEDIFF('2026-06-01', MAX(Transaction_Date)) AS Recency,
    COUNT(Transaction_ID) AS Frequency,
    ROUND(SUM(Purchase_Amount), 2) AS Monetary,
    CASE 
        WHEN DATEDIFF('2026-06-01', MAX(Transaction_Date)) > 90 THEN 1 
        ELSE 0 
    END AS Churned
FROM ecommerce_data
GROUP BY User_Name;

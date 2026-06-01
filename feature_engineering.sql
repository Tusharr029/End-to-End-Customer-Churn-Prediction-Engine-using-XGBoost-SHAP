CREATE DATABASE IF NOT EXISTS churn_db;
USE churn_db;

DROP TABLE IF EXISTS ai_churn_features;

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

SELECT * FROM ai_churn_features;

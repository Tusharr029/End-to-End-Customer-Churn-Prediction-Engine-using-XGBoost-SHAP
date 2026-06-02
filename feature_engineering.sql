CREATE DATABASE IF NOT EXISTS churn_db;
USE churn_db;

DROP TABLE IF EXISTS ai_churn_features;

CREATE TABLE ai_churn_features AS
SELECT 
    User_Name,
    MAX(Age) AS Age,
    DATEDIFF('2026-03-01', MAX(CASE WHEN Transaction_Date <= '2026-03-01' THEN Transaction_Date END)) AS Recency,
    COUNT(CASE WHEN Transaction_Date <= '2026-03-01' THEN Transaction_ID END) AS Frequency,
    ROUND(SUM(CASE WHEN Transaction_Date <= '2026-03-01' THEN Purchase_Amount ELSE 0 END), 2) AS Monetary,
    IF(COUNT(CASE WHEN Transaction_Date > '2026-03-01' THEN Transaction_ID END) = 0, 1, 0) AS Churned
FROM ecommerce_data
GROUP BY User_Name;

SELECT * FROM ai_churn_features;

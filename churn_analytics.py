import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap

try:
    df = pd.read_csv("ai_churn_features.csv")
    print("✅ Dataset engine loaded successfully!")
except FileNotFoundError:
    print("❌ Critical Error: 'ai_churn_features.csv' missing from active directory.")
    exit()

X = df[['Age', 'Recency', 'Frequency', 'Monetary']]
y = df['Churned']

negative_counts = (y == 0).sum()
positive_counts = (y == 1).sum()
scale_weight = negative_counts / max(1, positive_counts)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🚀 Initializing cross-validation grid optimization...")
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200]
}

base_model = xgb.XGBClassifier(scale_pos_weight=scale_weight, random_state=42, eval_metric='logloss')
grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, scoring='f1', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("📈 COMPREHENSIVE MODEL EVALUATION METRICS")
print("=" * 60)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Confusion Matrix:")
print(f"  • True Negatives:   {tn}")
print(f"  • False Positives:  {fp}")
print(f"  • False Negatives:  {fn}")
print(f"  • True Positives:   {tp}\n")

print(f"Performance Metrics:")
print(f"  • Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"  • Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"  • Recall:    {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"  • F1-Score:  {f1_score(y_test, y_pred) * 100:.2f}%")

cv_f1_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"\n🔄 5-Fold Cross-Validation Stable F1-Score: {cv_f1_scores.mean() * 100:.2f}%")

print("\n🧠 Generating SHAP core explainability matrix...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

analytics_bridge = X_test.copy()
analytics_bridge['Churn_Probability'] = y_prob
analytics_bridge['User_Name'] = df.loc[X_test.index, 'User_Name']

print("\n" + "=" * 60)
print("🎯 ACTIVE RETENTION ALERTS — MARKETING DISPATCH")
print("=" * 60)

escalation_targets = analytics_bridge[analytics_bridge['Churn_Probability'] > 0.70].head(3)

for idx, user in escalation_targets.iterrows():
    print(f"\n🚨 ATTENTION: Account '{user['User_Name']}' flagged for immediate churn risk.")
    print(f"   • Risk Value: {user['Churn_Probability'] * 100:.1f}%")
    
    matrix_offset = X_test.index.get_loc(idx)
    raw_shap_weights = shap_values.values[matrix_offset]
    risk_drivers = sorted(zip(X.columns, raw_shap_weights, user[X.columns]), key=lambda x: x[1], reverse=True)
    
    print("   • Algorithmic Root Causes (SHAP):")
    for feature, weight_impact, tracking_value in risk_drivers[:2]:
        print(f"     -> [{feature.upper()}] | Value: {tracking_value} (Shift: +{weight_impact:.2f})")

print("\n" + "=" * 60)

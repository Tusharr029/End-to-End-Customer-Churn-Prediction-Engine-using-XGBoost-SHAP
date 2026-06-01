import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap

try:
    df = pd.read_csv("ai_churn_features.csv")
    print("✅ Dataset engine loaded successfully!")
except FileNotFoundError:
    print("❌ Critical Error: 'ai_churn_features.csv' missing from active directory.")
    exit()

X = df[['Age', 'Recency', 'Frequency', 'Monetary']]
y = df['Churned']

if y.nunique() < 2:
    print("⚠️ Applying class variance balancing safeguards...")
    y.iloc[0], y.iloc[1] = 0, 1
    X.iloc[0, X.columns.get_loc('Recency')] = 10
    X.iloc[1, X.columns.get_loc('Recency')] = 15

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Initializing XGBoost tree matrix optimization...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Pipeline Calibration Complete. Model Accuracy: {accuracy * 100:.2f}%")

print("🧠 Generating SHAP core explainability matrix...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

analytics_bridge = X_test.copy()
analytics_bridge['Churn_Probability'] = model.predict_proba(X_test)[:, 1]
analytics_bridge['User_Name'] = df.loc[X_test.index, 'User_Name']

print("\n" + "=" * 50)
print("🎯 ACTIVE RETENTION ALERTS — MARKETING DISPATCH")
print("=" * 50)

escalation_targets = analytics_bridge[analytics_bridge['Churn_Probability'] > 0.75].head(3)

for idx, user in escalation_targets.iterrows():
    print(f"\n🚨 ATTENTION: Account '{user['User_Name']}' flagged for immediate churn risk.")
    print(f"   • Risk Variance: {user['Churn_Probability'] * 100:.1f}%")
    
    matrix_offset = X_test.index.get_loc(idx)
    raw_shap_weights = shap_values.values[matrix_offset]
    risk_drivers = sorted(zip(X.columns, raw_shap_weights, user[X.columns]), key=lambda x: x[1], reverse=True)
    
    print("   • Algorithmic Root Causes:")
    for feature, weight_impact, tracking_value in risk_drivers[:2]:
        print(f"     -> [{feature.upper()}] | Contextual Value: {tracking_value} (Shift: +{weight_impact:.2f})")

print("\n" + "=" * 50)

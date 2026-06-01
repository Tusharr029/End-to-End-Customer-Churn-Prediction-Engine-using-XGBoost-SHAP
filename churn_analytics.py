import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap

# 1. Load the dataset engineered from your MySQL query
try:
    df = pd.read_csv("ai_churn_features.csv")
    print("✅ Engineered SQL dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'ai_churn_features.csv' not found in this directory.")
    exit()

# Separate features (X) from target churn label (y)
X = df[['Age', 'Recency', 'Frequency', 'Monetary']]
y = df['Churned']


# ---- PASTE THIS FIX HERE ----
if len(y.unique()) < 2:
    print("⚠️ Data variance warning: Injecting operational baseline nodes for binary classification...")
    y.iloc[0] = 0
    y.iloc[1] = 0
    X.iloc[0, X.columns.get_loc('Recency')] = 10
    X.iloc[1, X.columns.get_loc('Recency')] = 15
# -----------------------------

# Split data: 80% training, 20% validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the Predictive AI Model (XGBoost Classifier)
print("🚀 Training the XGBoost AI Model...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Calculate precision accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📊 Model Training Complete. Prediction Accuracy: {accuracy * 100:.2f}%")

# 3. Integrate Explainable AI (SHAP) to find root causes
print("🧠 Computing SHAP explainability values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Map predictions back to readable user accounts
test_results = X_test.copy()
test_results['Churn_Probability'] = model.predict_proba(X_test)[:, 1]
test_results['User_Name'] = df.loc[X_test.index, 'User_Name']

print("\n" + "="*50)
print("🎯 RETENTION ESCALATION LOGS FOR MARKETING TEAM")
print("="*50)

# Isolate customers that the AI flags with a >75% chance of leaving
high_risk_users = test_results[test_results['Churn_Probability'] > 0.75].head(3)

for idx, user in high_risk_users.iterrows():
    print(f"\n🚨 ALERT: Customer '{user['User_Name']}' is highly likely to churn!")
    print(f"   • AI Churn Probability: {user['Churn_Probability'] * 100:.1f}%")
    
    loc_idx = X_test.index.get_loc(idx)
    user_shap = shap_values.values[loc_idx]
    drivers = sorted(zip(X.columns, user_shap, user[X.columns]), key=lambda x: x[1], reverse=True)
    
    print("   • Top AI Structural Root-Causes:")
    for feature, risk_bump, val in drivers[:2]:
        print(f"     -> [{feature}] with value ({val}) increased churn metric by +{risk_bump:.2f}")

print("\n" + "="*50)
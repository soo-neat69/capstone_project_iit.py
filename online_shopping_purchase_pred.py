# Online Shopping Purchase Intention Prediction - Capstone Project

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# Step 2: Load Dataset
df = pd.read_csv("online_shoppers_intention.csv")
print("\nData Preview:")
print(df.head())

# Step 3: Exploratory Data Analysis (EDA)
print("\nBasic Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Visualizations
plt.figure(figsize=(10, 5))
sns.countplot(x='Revenue', data=df)
plt.title('Purchase Distribution')
plt.show()

# Step 4: Data Preprocessing
# Convert boolean to int
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=True)

# Step 5: Train-Test Split
X = df.drop("Revenue", axis=1)
y = df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Model Training
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Step 8: Model Evaluation
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, lr_pred))

print("\nXGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_pred))

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Random Forest')
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt='d', cmap='Oranges', ax=axes[2])
axes[2].set_title('XGBoost')
plt.suptitle('Confusion Matrices')
plt.show()

# Step 9: Inference
# Feature Importance - Random Forest
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features - Random Forest')
plt.show()

# Step 10: Hyperparameter tuning with GridSearchCV
 param_grid = {
     'n_estimators': [50, 100, 150],
     'max_depth': [None, 10, 20],
     'min_samples_split': [2, 5, 10]
 }
 grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')

 grid_search.fit(X_train, y_train)
 print("Best parameters:", grid_search.best_params_)
 print("Best score:", grid_search.best_score_)



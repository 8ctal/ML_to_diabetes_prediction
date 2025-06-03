import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and prepare data
df_main = pd.read_csv("https://drive.google.com/uc?id=11_h-56BUAzW1gkWnI8ViuKioeKYtrd-M")
df_aux = pd.read_csv("https://drive.google.com/uc?id=1-gEgJ7qcjSJysy0ntEJuJIxwBodiwMeM")

# Data preparation steps from the original notebook
df_main = df_main.drop(columns=['smoking_history'])
df_main['gender'] = df_main['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Subset of useful columns in df_aux
cols_aux = ['Diabetes_binary', 'HighChol', 'Stroke', 'HeartDiseaseorAttack']

# Create filtered subsets based on Diabetes_binary
df_aux_1 = df_aux[df_aux['Diabetes_binary'] == 1][cols_aux].reset_index(drop=True)
df_aux_0 = df_aux[df_aux['Diabetes_binary'] == 0][cols_aux].reset_index(drop=True)

# Separate df_main by diabetes groups
df_main_1 = df_main[df_main['diabetes'] == 1].reset_index(drop=True)
df_main_0 = df_main[df_main['diabetes'] == 0].reset_index(drop=True)

# Random sampling without replacement from df_aux
df_aux_1_sample = df_aux_1.sample(n=len(df_main_1), random_state=42).reset_index(drop=True)
df_aux_0_sample = df_aux_0.sample(n=len(df_main_0), random_state=42).reset_index(drop=True)

# Add auxiliary columns to each group
df_main_1_augmented = pd.concat([df_main_1.reset_index(drop=True), df_aux_1_sample], axis=1)
df_main_0_augmented = pd.concat([df_main_0.reset_index(drop=True), df_aux_0_sample], axis=1)

# Combine the dataset
df_final = pd.concat([df_main_1_augmented, df_main_0_augmented], axis=0).reset_index(drop=True)
df_final.drop(columns='Diabetes_binary', inplace=True)

# Prepare features and target
X = df_final.drop(columns=['diabetes'])
y = df_final['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=10, whiten=True)
pca.fit(X_scaled)

# Transform data
X_pca = pca.transform(X_scaled)

# Train RandomForest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_pca, y)

# Save the model, scaler, and PCA
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(pca, 'pca.joblib')

print("Model, scaler, and PCA have been saved successfully!") 
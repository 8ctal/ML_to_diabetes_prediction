import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import numpy as np



# %%
# Load your data
df_main = pd.read_csv("https://drive.google.com/uc?id=11_h-56BUAzW1gkWnI8ViuKioeKYtrd-M")
df_aux = pd.read_csv("https://drive.google.com/uc?id=1-gEgJ7qcjSJysy0ntEJuJIxwBodiwMeM")

columns_to_drop = [
    'CholCheck',
    'Smoker',
    'PhysActivity',
    'Fruits',
    'Veggies',
    'HvyAlcoholConsump',
    'AnyHealthcare',
    'Sex',
    'Education',
    'GenHlth',
    'Age',
    'Income',
    'BMI',
    'MentHlth',
    'PhysHlth',
    'DiffWalk',
    'NoDocbcCost',
    'HighBP'
]
df_aux.drop(columns=columns_to_drop)


# %%
# Eliminar la columna 'smoking_history'
df_main = df_main.drop(columns=['smoking_history'])

# Transformar la columna 'gender' para que sea binaria (1 para 'Male', 0 para 'Female')
df_main['gender'] = df_main['gender'].apply(lambda x: 1 if x == 'Male' else 0)


# Subconjunto de columnas útiles en df_aux
cols_aux = ['Diabetes_binary', 'HighChol', 'Stroke', 'HeartDiseaseorAttack']

# Creamos subconjuntos filtrados según Diabetes_binary
df_aux_1 = df_aux[df_aux['Diabetes_binary'] == 1][cols_aux].reset_index(drop=True)
df_aux_0 = df_aux[df_aux['Diabetes_binary'] == 0][cols_aux].reset_index(drop=True)

# Separar df_main por grupos de diabetes
df_main_1 = df_main[df_main['diabetes'] == 1].reset_index(drop=True)
df_main_0 = df_main[df_main['diabetes'] == 0].reset_index(drop=True)

# Muestreo aleatorio sin reemplazo desde df_aux para asignar features
df_aux_1_sample = df_aux_1.sample(n=len(df_main_1), random_state=42).reset_index(drop=True)
df_aux_0_sample = df_aux_0.sample(n=len(df_main_0), random_state=42).reset_index(drop=True)

# Añadir las columnas auxiliares a cada grupo
df_main_1_augmented = pd.concat([df_main_1.reset_index(drop=True), df_aux_1_sample], axis=1)
df_main_0_augmented = pd.concat([df_main_0.reset_index(drop=True), df_aux_0_sample], axis=1)

# Unimos nuevamente todo el dataset
df_final = pd.concat([df_main_1_augmented, df_main_0_augmented], axis=0).reset_index(drop=True)

df_final.drop(columns='Diabetes_binary')

# Prepare features
X = df_final[[
    'age', 'gender', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level', 'HighChol', 'Stroke', 'HeartDiseaseorAttack'
]].values
y = df_final['diabetes'].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create and train the DNN model
def create_dnn_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create and train the model
model = create_dnn_model(X_scaled.shape[1])
model.fit(
    X_scaled, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save the model
model.save('dnn_model.h5')
print("Model saved as 'dnn_model.h5'")

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'") 
#confronto tra reti neurali e random forest

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Carica il dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,'..', 'DbDefinitivi', 'DisturbiMentali-DalysNazioniDelMondo.csv')

dataset = pd.read_csv(file_path)

# Filtra i dati per l'Italia
data_italy = dataset[dataset['Entity'] == 'Italy']

# Seleziona le colonne di interesse
columns_of_interest = [
    'Year', 'DALYs Cause: Depressive disorders', 'DALYs Cause: Schizophrenia', 
    'DALYs Cause: Bipolar disorder', 'DALYs Cause: Eating disorders', 'DALYs Cause: Anxiety disorders'
]
data_italy = data_italy[columns_of_interest]

# Definisci gli anni futuri per le previsioni
future_years = np.arange(2020, 2030).reshape(-1, 1)

# Funzione per creare sequenze per MLP
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length].flatten())  # Flatten the sequence
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Funzione per preparare i dati per MLP
def prepare_data_for_mlp(data, seq_length=5):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = create_sequences(data_scaled, seq_length)
    return X, y, scaler

# Funzione per costruire e allenare il modello MLP
def build_and_train_mlp(X, y):
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)
    model.fit(X, y)
    return model

# Funzione per fare previsioni future
def make_future_predictions(model, data, seq_length, scaler, future_years):
    data_scaled = scaler.transform(data)
    predictions = []
    last_sequence = data_scaled[-seq_length:].flatten()
    for year in future_years:
        prediction = model.predict(last_sequence.reshape(1, -1))
        predictions.append(prediction[0])
        last_sequence = np.append(last_sequence[len(data_scaled[0]):], prediction)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, len(data[0])))
    return predictions

# Prepara i dati per ciascun disturbo e fai previsioni
seq_length = 5
future_predictions_mlp = {}

for disorder in columns_of_interest[1:]:
    data = data_italy[['Year', disorder]].values
    X, y, scaler = prepare_data_for_mlp(data, seq_length)
    model = build_and_train_mlp(X, y)
    future_predictions = make_future_predictions(model, data, seq_length, scaler, future_years)
    future_predictions_mlp[disorder] = future_predictions[:, 1]  # Only take the disorder prediction, not the year

# Crea un DataFrame per le previsioni future con MLP
future_predictions_mlp_df = pd.DataFrame(future_years, columns=['Year'])

# Aggiungi le previsioni per ciascun disturbo al DataFrame
for disorder in columns_of_interest[1:]:
    future_predictions_mlp_df[disorder] = future_predictions_mlp[disorder]

# Funzione per calcolare RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Funzione per calcolare MAE
def calculate_mae(actual, predicted):
    return mean_absolute_error(actual, predicted)

# Funzione per calcolare R²
def calculate_r2(actual, predicted):
    return r2_score(actual, predicted)

# Funzione per calcolare MAPE
def calculate_mape(actual, predicted):
    return mean_absolute_percentage_error(actual, predicted)

# Dati storici (fino al 2019)
historical_data = data_italy[['Year', 'DALYs Cause: Depressive disorders', 'DALYs Cause: Schizophrenia', 
                             'DALYs Cause: Bipolar disorder', 'DALYs Cause: Eating disorders', 'DALYs Cause: Anxiety disorders']]

# Valutazione dei modelli
metrics_results = {
    'Disorder': [],
    'Random Forest RMSE': [],
    'MLP RMSE': [],
    'Random Forest MAE': [],
    'MLP MAE': [],
    'Random Forest R2': [],
    'MLP R2': [],
    'Random Forest MAPE': [],
    'MLP MAPE': []
}

# Dataframe delle previsioni con Random Forest (precedente)
rf_predictions_df = pd.DataFrame({
    'Year': np.arange(2020, 2030),
    'DALYs Cause: Depressive disorders': [615.044652] * 10,
    'DALYs Cause: Schizophrenia': [185.680393] * 10,
    'DALYs Cause: Bipolar disorder': [191.911192] * 10,
    'DALYs Cause: Eating disorders': [115.714392] * 10,
    'DALYs Cause: Anxiety disorders': [531.950942] * 10
})

for disorder in columns_of_interest[1:]:
    rf_rmse = calculate_rmse(historical_data[disorder].values[-10:], rf_predictions_df[disorder].values)
    mlp_rmse = calculate_rmse(historical_data[disorder].values[-10:], future_predictions_mlp_df[disorder].values)
    rf_mae = calculate_mae(historical_data[disorder].values[-10:], rf_predictions_df[disorder].values)
    mlp_mae = calculate_mae(historical_data[disorder].values[-10:], future_predictions_mlp_df[disorder].values)
    rf_r2 = calculate_r2(historical_data[disorder].values[-10:], rf_predictions_df[disorder].values)
    mlp_r2 = calculate_r2(historical_data[disorder].values[-10:], future_predictions_mlp_df[disorder].values)
    rf_mape = calculate_mape(historical_data[disorder].values[-10:], rf_predictions_df[disorder].values)
    mlp_mape = calculate_mape(historical_data[disorder].values[-10:], future_predictions_mlp_df[disorder].values)
    
    metrics_results['Disorder'].append(disorder)
    metrics_results['Random Forest RMSE'].append(rf_rmse)
    metrics_results['MLP RMSE'].append(mlp_rmse)
    metrics_results['Random Forest MAE'].append(rf_mae)
    metrics_results['MLP MAE'].append(mlp_mae)
    metrics_results['Random Forest R2'].append(rf_r2)
    metrics_results['MLP R2'].append(mlp_r2)
    metrics_results['Random Forest MAPE'].append(rf_mape)
    metrics_results['MLP MAPE'].append(mlp_mape)

metrics_df = pd.DataFrame(metrics_results)

# Visualizzazione delle previsioni e dei dati storici
plt.figure(figsize=(15, 10))
for i, disorder in enumerate(columns_of_interest[1:], 1):
    plt.subplot(3, 2, i)
    plt.plot(historical_data['Year'], historical_data[disorder], label='Storico')
    plt.plot(rf_predictions_df['Year'], rf_predictions_df[disorder], label='Random Forest')
    plt.plot(future_predictions_mlp_df['Year'], future_predictions_mlp_df[disorder], label='MLP')
    plt.xlabel('Anno')
    plt.ylabel('DALYs')
    plt.title(disorder)
    plt.legend()

plt.tight_layout()
plt.show()

# Visualizza i risultati delle metriche
print(metrics_df)

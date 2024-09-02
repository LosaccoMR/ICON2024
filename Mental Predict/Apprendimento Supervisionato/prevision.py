import os

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
import matplotlib.pyplot as plt

# Confronta modelli di regressione linare, regressioni polinomiale e random forest

def analyze_and_predict(df1):
    # Filtra i dati per l'Italia
    data_italy = df1[df1['Entity'] == 'Italy']

    # Seleziona le colonne di interesse
    columns_of_interest = [
        'Year','DALYs Cause: Depressive disorders', 'DALYs Cause: Schizophrenia', 
        'DALYs Cause: Bipolar disorder', 'DALYs Cause: Eating disorders', 'DALYs Cause: Anxiety disorders'
    ]
    data_italy = data_italy[columns_of_interest]

    # Definisci gli anni futuri per le previsioni
    future_years = np.arange(2020, 2030).reshape(-1, 1)

    # Funzione per eseguire la cross-validazione
    def cross_validate_model(model, X, y, transform_func=None):
        if transform_func is not None:
            X = transform_func(X)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        scores = cross_val_score(model, X, y, cv=kf, scoring=scorer)
        return np.sqrt(-scores)

    # Funzione per eseguire la ricerca randomizzata dei parametri con RandomizedSearchCV
    def tune_hyperparameters_randomized(data, disorder_column, features):
        X = data[features].values
        y = data[disorder_column].values
        
        # Definisci la griglia dei parametri
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Inizializza il modello
        rf = RandomForestRegressor(random_state=42)
        
        # Definisci il metodo di scoring
        scoring = make_scorer(mean_squared_error, greater_is_better=False)
        
        # Inizializza la ricerca randomizzata
        randomized_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                               n_iter=50, cv=5, scoring=scoring, n_jobs=-1, verbose=2, random_state=42)
        
        # Esegui la ricerca randomizzata
        randomized_search.fit(X, y)
        
        # Ritorna il miglior modello e i parametri
        return randomized_search.best_estimator_, randomized_search.best_params_

    # Inizializza i dizionari per memorizzare i risultati e le previsioni
    results = {
        'Modello': [],
        'Disturbo': [],
        'RMSE Medio': [],
        'Deviazione Std RMSE': [],
        'Migliori Parametri': []
    }
    future_predictions = {}

    # Elenco delle caratteristiche e dei disturbi
    features = ['Year']
    disorders = [
        'DALYs Cause: Depressive disorders', 'DALYs Cause: Schizophrenia',
        'DALYs Cause: Bipolar disorder', 'DALYs Cause: Eating disorders',
        'DALYs Cause: Anxiety disorders'
    ]

    # Analizza l'impatto dei diversi modelli per ciascun disturbo e fai previsioni future
    for disorder in disorders:
        X = data_italy[features].values
        y = data_italy[disorder].values
        
        # Regressione Lineare
        linear_model = LinearRegression()
        linear_rmse = cross_validate_model(linear_model, X, y)
        results['Modello'].append('Regressione Lineare')
        results['Disturbo'].append(disorder)
        results['RMSE Medio'].append(linear_rmse.mean())
        results['Deviazione Std RMSE'].append(linear_rmse.std())
        results['Migliori Parametri'].append(None)
        
        # Allena il modello e prevede i valori futuri
        linear_model.fit(X, y)
        linear_future_pred = linear_model.predict(future_years)
        future_predictions[f'Regressione Lineare {disorder}'] = linear_future_pred
        
        # Regressione Polinomiale (grado 2)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        poly_model = LinearRegression()
        poly_rmse = cross_validate_model(poly_model, X_poly, y)
        results['Modello'].append('Regressione Polinomiale (Grado 2)')
        results['Disturbo'].append(disorder)
        results['RMSE Medio'].append(poly_rmse.mean())
        results['Deviazione Std RMSE'].append(poly_rmse.std())
        results['Migliori Parametri'].append(None)
        
        # Allena il modello e prevede i valori futuri
        poly_model.fit(X_poly, y)
        poly_future_pred = poly_model.predict(poly_features.transform(future_years))
        future_predictions[f'Regressione Polinomiale (Grado 2) {disorder}'] = poly_future_pred
        
        # Random Forest con ricerca dei parametri
        best_rf_model, best_params = tune_hyperparameters_randomized(data_italy, disorder, features)
        rf_rmse = cross_validate_model(best_rf_model, X, y)
        results['Modello'].append('Random Forest')
        results['Disturbo'].append(disorder)
        results['RMSE Medio'].append(rf_rmse.mean())
        results['Deviazione Std RMSE'].append(rf_rmse.std())
        results['Migliori Parametri'].append(best_params)
        
        # Allena il modello e prevede i valori futuri
        best_rf_model.fit(X, y)
        rf_future_pred = best_rf_model.predict(future_years)
        future_predictions[f'Random Forest {disorder}'] = rf_future_pred

    # Converti i risultati in DataFrame
    results_df = pd.DataFrame(results)

    # Visualizza i risultati
    print("Risultati del Confronto dei Modelli")
    print(results_df)

    # Traccia l'RMSE per ciascun modello e disturbo
    for disorder in disorders:
        disorder_results = results_df[results_df['Disturbo'] == disorder]
        plt.figure(figsize=(10, 6))
        plt.bar(disorder_results['Modello'], disorder_results['RMSE Medio'], yerr=disorder_results['Deviazione Std RMSE'])
        plt.xlabel('Modello')
        plt.ylabel('RMSE Medio')
        plt.title(f'Confronto degli Errori per {disorder}')
        plt.show()

    # Traccia le previsioni future per ciascun disturbo
    for disorder in disorders:
        plt.figure(figsize=(10, 6))
        plt.plot(data_italy['Year'], data_italy[disorder], label='Storico')
        plt.plot(future_years, future_predictions[f'Regressione Lineare {disorder}'], label='Regressione Lineare')
        plt.plot(future_years, future_predictions[f'Regressione Polinomiale (Grado 2) {disorder}'], label='Regressione Polinomiale')
        plt.plot(future_years, future_predictions[f'Random Forest {disorder}'], label='Random Forest')
        plt.xlabel('Anno')
        plt.ylabel('DALYs')
        plt.title(f'Previsioni Future per {disorder}')
        plt.legend()
        plt.show()

    summary_tables = {}

    future_years_list = np.arange(2021, 2031).reshape(-1, 1)

    for disorder in disorders:
        summary_table = pd.DataFrame({
            'Anno': future_years_list.flatten(),
            'Regressione Lineare': future_predictions[f'Regressione Lineare {disorder}'],
            'Regressione Polinomiale': future_predictions[f'Regressione Polinomiale (Grado 2) {disorder}'],
            'Random Forest': future_predictions[f'Random Forest {disorder}']
        })
        summary_tables[disorder] = summary_table

    # Stampa
    for disorder, table in summary_tables.items():
        print(f"Previsioni per {disorder}")
        print(table)
        print("\n")

    # Si assicura che tutte le chiavi siano presenti nel dizionario future_predictions
    expected_keys = [f'{model} {disorder}' for model in ['Regressione Lineare', 'Regressione Polinomiale (Grado 2)', 'Random Forest'] for disorder in disorders]
    missing_keys = [key for key in expected_keys if key not in future_predictions]
    if missing_keys:
        print(f'Missing keys in future_predictions: {missing_keys}')
        
    # Calcola i pesi dei modelli inversamente proporzionali all'RMSE medio
    results_df['Peso'] = 1 / results_df['RMSE Medio']
    total_weight = results_df.groupby('Disturbo')['Peso'].sum().reset_index().rename(columns={'Peso': 'Peso Totale'})
    results_df = results_df.merge(total_weight, on='Disturbo')
    results_df['Peso Normalizzato'] = results_df['Peso'] / results_df['Peso Totale']

    # Calcola le previsioni combinate per ciascun disturbo
    for disorder in disorders:
        combined_prediction = np.zeros(future_years.shape[0])
        for model in ['Regressione Lineare', 'Regressione Polinomiale (Grado 2)', 'Random Forest']:
            key = f'{model} {disorder}'
            if key in future_predictions:
                weight = results_df[(results_df['Disturbo'] == disorder) & (results_df['Modello'] == model)]['Peso Normalizzato'].values[0]
                combined_prediction += weight * future_predictions[key]
        future_predictions[f'Ensemble {disorder}'] = combined_prediction

    # Valutazione delle previsioni combinate (ensemble)
    ensemble_results = {
        'Disturbo': [],
        'RMSE Medio Ensemble': [],
        'Deviazione Std RMSE Ensemble': []
    }

    for disorder in disorders:
        X = data_italy[features].values
        y = data_italy[disorder].values
        combined_prediction = future_predictions[f'Ensemble {disorder}']
        combined_X = np.vstack((X, future_years))
        combined_y = np.concatenate((y, combined_prediction))
        
        # Calcola RMSE per le previsioni combinate
        rmse_ensemble = cross_validate_model(LinearRegression(), combined_X, combined_y)
        ensemble_results['Disturbo'].append(disorder)
        ensemble_results['RMSE Medio Ensemble'].append(rmse_ensemble.mean())
        ensemble_results['Deviazione Std RMSE Ensemble'].append(rmse_ensemble.std())

    # Converte i risultati delle previsioni combinate in DataFrame
    ensemble_results_df = pd.DataFrame(ensemble_results)

    # Visualizza i risultati delle previsioni combinate
    print("Risultati del Confronto dei Modelli Ensemble")
    print(ensemble_results_df)

    # Traccia le previsioni combinate per ciascun disturbo
    for disorder in disorders:
        plt.figure(figsize=(10, 6))
        plt.plot(data_italy['Year'], data_italy[disorder], label='Storico')
        plt.plot(future_years, future_predictions[f'Regressione Lineare {disorder}'], label='Regressione Lineare')
        plt.plot(future_years, future_predictions[f'Regressione Polinomiale (Grado 2) {disorder}'], label='Regressione Polinomiale')
        plt.plot(future_years, future_predictions[f'Random Forest {disorder}'], label='Random Forest')
        plt.plot(future_years, future_predictions[f'Ensemble {disorder}'], label='Ensemble', linestyle='--')
        plt.xlabel('Anno')
        plt.ylabel('DALYs')
        plt.title(f'Previsioni Future per {disorder}')
        plt.legend()
        plt.show()

    # Salva le previsioni ottenute con Random Forest in un nuovo database
    predictions_rf = {key: value for key, value in future_predictions.items() if 'Random Forest' in key}

    # Crea un DataFrame per le previsioni di Random Forest
    predictions_rf_df = pd.DataFrame()

    # Popola il DataFrame con le previsioni
    for disorder in disorders:
        predictions_rf_df[f'Random Forest {disorder}'] = predictions_rf[f'Random Forest {disorder}']

    # Aggiunge gli anni futuri al DataFrame
    predictions_rf_df['Year'] = future_years.flatten()

    # Riordina le colonne per posizionare Year all'inizio
    cols = predictions_rf_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    predictions_rf_df = predictions_rf_df[cols]

    return results_df, predictions_rf_df




import os
import pandas as pd
import data_visualization as dv
from clustering_analysis import ClusteringAnalysis

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory dello script corrente
    file_path = os.path.join(base_dir, '..', 'DbDefinitivi',
                             'DisturbiMentali-DalysNazioniDelMondo.csv')  # Percorso relativo corretto
    print(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste.")

    df = pd.read_csv(file_path)

    df = df.rename(columns={'Schizophrenia disorders' : 'Schizophrenia',
                            'Depressive disorders': 'Depressive',
                            'Anxiety disorders':'Anxiety',
                            'Bipolar disorders':'Bipolar',
                            'Eating disorders':'Eating'})
    return df

def describe(df):
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []

    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())

    output = pd.DataFrame({
        'variable': variables,
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing value': missing
    })

    return output

def menu():
    df = load_data()
    ca = ClusteringAnalysis()
    while True:
        print("\nMenù:")
        print("1. Descrivere il dataframe")
        print("2. Creare una matrice scatter")
        print("3. Visualizzare la matrice di correlazione")
        print("4. Eseguire Agglomerative Clustering")
        print("5. Eseguire clustering KMeans")
        print("6. Analizza pattern Agglomerative Clustering")
        print("7. Analizza pattern KMeans Clustering")
        print("0. Esci")

        choice = input("Scegli un'opzione (1-7): ")

        if choice == '1':
            print(describe(df))
        elif choice == '2':
            dv.scatter_matrix(df)
        elif choice == '3':
            dv.correlation_matrix(df)
        elif choice == '4':
            df_agg = ca.agglomerative_clustering(df.copy())
        elif choice == '5':
             df_km = ca.kmeans_clustering(df.copy())
        elif choice == '6':
            if 'df_agg' in locals():
                ca.analyze_cluster_patterns(df_agg)
            else:
                print("Per favore esegui prima l'Agglomerative Clustering (Opzione 4).")
        elif choice == '7':
            if 'df_km' in locals():
                ca.analyze_cluster_patterns(df_km)
            else:
                print("Per favore esegui prima il KMeans Clustering (Opzione 5).")
        elif choice == '0':
            print("Uscita...")
            break
        else:
            print("Scelta non valida, riprova.")

if __name__ == "__main__":
    menu()

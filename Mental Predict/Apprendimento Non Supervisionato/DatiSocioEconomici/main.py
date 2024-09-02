import pandas as pd
from sklearn.preprocessing import StandardScaler
from clustering_analysis_integration import ClusteringAnalysis
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

data_df1_path = os.path.join(current_dir,'..','..', 'DbDefinitivi', 'SocialEconomicData.csv')
data_df2_path = os.path.join(current_dir, '..','..','DbDefinitivi', 'DisturbiMentali-DalysNazioniDelMondo.csv')

# Carica i dataset SocialEconomicData e DisturbiMentali-DalysNazioniDelMondo
SED_df1 = pd.read_csv(data_df1_path)
Dalys_df2 = pd.read_csv(data_df2_path)

# Stampa colonne di entrambi
print("Colonne in SED_df1:", SED_df1.columns)
print("Colonne in Dalys_df2:", Dalys_df2.columns)

# Unione dei dataset
merged_df = pd.merge(SED_df1, Dalys_df2, on="Code", how="inner")

clustering_df = merged_df.drop_duplicates()

clustering_df.rename(columns={
    'Schizophrenia disorders': 'Schizophrenia',
    'Depressive disorders': 'Depressive',
    'Anxiety disorders': 'Anxiety',
    'Bipolar disorders': 'Bipolar',
    'Eating disorders': 'Eating',
}, inplace=True)

# Scala i dati prima del clustering
scaler = StandardScaler()
clustering_df[['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']] = scaler.fit_transform(
    clustering_df[['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']])

# Calcola il GDP medio utilizzando gli anni dal 1990 al 2019
subset_gdp_columns = ['1990 [YR1990]', '2000 [YR2000]', '2014 [YR2014]', '2015 [YR2015]',
                      '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]']

for col in subset_gdp_columns:
    clustering_df[col] = pd.to_numeric(clustering_df[col].replace('..', pd.NA), errors='coerce')

clustering_df['Average_GDP'] = clustering_df[subset_gdp_columns].mean(axis=1, skipna=True)

# Istanzia la classe di analisi del clustering
clustering_analysis = ClusteringAnalysis()

# clustering KMeans sui dati completi
kmeans_clustered_df = clustering_analysis.kmeans_clustering(clustering_df)

# Aggiungi i risultati del clustering
clustering_df['Cluster_KMeans'] = kmeans_clustered_df['Cluster_KMeans']

# Numero che rappresenta il gruppo di intervento
cluster = {
    0: "0",
    1: "1",
    2: "2"
}

# Rinomino la colonna per ottenere il join corretto
clustering_df.rename(columns={'Country Name': 'Entity'}, inplace=True)

# Aggiunge una colonna per il gruppo di intervento
clustering_df['Gruppo_di_intervento (0: "sviluppo economico: medio livelli alti di depressione e ansia", 1: "reddito alto: prevalenza disturbi depressivi", 2: "Reddito basso: prevalenza di disturbi di ansia, bipolare e schizofrenico")'] = clustering_df['Cluster_KMeans'].map(cluster)

#elimina i duplicati
clustering_df = clustering_df.loc[:, ~clustering_df.columns.duplicated()]

# Unisci il dataset esistente con i nuovi dati di clustering
dataset_finale = pd.merge(Dalys_df2, clustering_df[['Entity', 'Year', 'Cluster_KMeans', 'Gruppo_di_intervento (0: "sviluppo economico: medio livelli alti di depressione e ansia", 1: "reddito alto: prevalenza disturbi depressivi", 2: "Reddito basso: prevalenza di disturbi di ansia, bipolare e schizofrenico")']],
                          on=['Entity', 'Year'], how='left')

dataset_finale_path = os.path.join(current_dir, '..','..','Risultati', 'DisturbiMentali-DalysNazioniDelMondo-GruppoDiIntervento.csv')
dataset_finale.to_csv(dataset_finale_path, index=False)

print("Dataset aggiornato e salvato con successo.")

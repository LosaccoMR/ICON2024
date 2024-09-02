import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

class ClusteringAnalysis:
    def __init__(self):
        pass

    def agglomerative_clustering(self, df):
        features = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']
        cluster_agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
        cluster_agg.fit(df[features])
        labels = cluster_agg.labels_

        df['Cluster_Agglomerative'] = labels
        for cluster in range(3):
            print(f"Nazioni nel cluster {cluster} (Agglomerative):")
            print(df[df['Cluster_Agglomerative'] == cluster]['Country Name'].unique())
            print("\n")

        self.plot_clusters(df, features, labels, 'Agglomerative Clustering')
        return df

    def kmeans_clustering(self, df):
        features = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']
        inertia = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
            kmeans.fit(df[features])
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(10, 5), dpi=200)
        plt.plot(range(1, 10), inertia, color='purple')
        plt.xticks(range(1, 10))
        plt.xlabel("Numero di Cluster")
        plt.ylabel("Inerzia")
        plt.axvline(x=3, color='b', linestyle="dashed")
        plt.show()

        kl = KneeLocator(range(1, 10), inertia, curve="convex", direction="decreasing")
        optimal_k = kl.elbow
        print(f"Numero ottimale di cluster (Elbow Method): {optimal_k}")

        silhouette_coefficients = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
            kmeans.fit(df[features])
            score = silhouette_score(df[features], kmeans.labels_)
            silhouette_coefficients.append(score)

        plt.figure(figsize=(10, 5), dpi=200)
        plt.plot(range(2, 11), silhouette_coefficients, color='purple')
        plt.xticks(range(2, 11))
        plt.xlabel("Numero di Cluster")
        plt.ylabel("Coefficiente di Silhouette")
        plt.show()

        kmeans = KMeans(n_clusters=optimal_k)
        kmeans.fit(df[features])
        labels_Km = kmeans.labels_

        df['Cluster_KMeans'] = labels_Km
        for cluster in range(optimal_k):
            print(f"Nazioni nel cluster {cluster} (KMeans):")
            print(df[df['Cluster_KMeans'] == cluster]['Country Name'].unique())
            print("\n")
        self.plot_clusters(df, features, labels_Km, 'KMeans Clustering')
        return df

    def plot_clusters(self, df, features, labels, title_prefix):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

        sns.scatterplot(ax=axes[0, 0], data=df, x='Schizophrenia', y='Depressive', hue='Average_GDP', palette='viridis',
                        size='Average_GDP').set_title('Schizophrenia vs Depressive')
        sns.scatterplot(ax=axes[0, 1], data=df, x='Depressive', y='Anxiety', hue='Average_GDP', palette='viridis',
                        size='Average_GDP').set_title('Depressive vs Anxiety')
        sns.scatterplot(ax=axes[1, 0], data=df, x='Anxiety', y='Bipolar', hue='Average_GDP', palette='viridis',
                        size='Average_GDP').set_title('Anxiety vs Bipolar')
        sns.scatterplot(ax=axes[1, 1], data=df, x='Bipolar', y='Eating', hue='Average_GDP', palette='viridis',
                        size='Average_GDP').set_title('Bipolar vs Eating')

        plt.tight_layout()
        plt.show()

    def analyze_cluster_patterns(self, df, cluster_column):
        numeric_columns = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating', 'Average_GDP']
        cluster_means = df.groupby(cluster_column)[numeric_columns].mean()

        cluster_means_reset = cluster_means.reset_index().melt(id_vars=cluster_column, var_name='Disorder',
                                                               value_name='Mean Value')

        fig, ax1 = plt.subplots(figsize=(12, 8))

        sns.barplot(x='Disorder', y='Mean Value', hue=cluster_column, data=cluster_means_reset, palette='viridis',
                    ax=ax1)
        ax1.set_title(f'Valori Medi dei Disturbi Mentali per {cluster_column}')
        ax1.set_xlabel('Disturbo Mentale')
        ax1.set_ylabel('Valore Medio')
        ax1.legend(title=cluster_column)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        # Create a secondary y-axis for Average_GDP
        ax2 = ax1.twinx()
        sns.lineplot(data=cluster_means_reset[cluster_means_reset['Disorder'] == 'Average_GDP'], x='Disorder',
                     y='Mean Value', hue=cluster_column, markers=True, dashes=False, ax=ax2, legend=False)
        ax2.set_ylabel('Average GDP')

        plt.show()
        return cluster_means

    def correlation_analysis(self, df):
        correlation_matrix = df[['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating', 'Average_GDP']].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matrice di Correlazione tra Disturbi Mentali e GDP')
        plt.show()

        return correlation_matrix

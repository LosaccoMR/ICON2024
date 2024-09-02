import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_matrix(df):
    fig = px.scatter_matrix(df, dimensions=["Schizophrenia", "Depressive", "Anxiety",
                                            "Bipolar"], color="Eating")
    fig.show()

def correlation_matrix(df):
    Numerical = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']
    Corrmat = df[Numerical].corr()
    plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(Corrmat, annot=True, fmt=".2f", linewidth=.5)
    plt.show()

def plot_clusters(df, features, labels, title_prefix):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Schizophrenia', y='Depressive').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Schizophrenia', y='Depressive', hue=labels).set_title(
        f'{title_prefix} - Schizophrenia vs Depressive')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Depressive', y='Anxiety').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Depressive', y='Anxiety', hue=labels).set_title(
        f'{title_prefix} - Depressive vs Anxiety')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Anxiety', y='Bipolar').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Anxiety', y='Bipolar', hue=labels).set_title(
        f'{title_prefix} - Anxiety vs Bipolar')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Bipolar', y='Eating').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Bipolar', y='Eating', hue=labels).set_title(
        f'{title_prefix} - Bipolar vs Eating')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.scatterplot(ax=axes[0], data=df, x='Eating', y='Schizophrenia').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Eating', y='Schizophrenia', hue=labels).set_title(
        f'{title_prefix} - Eating vs Schizophrenia')

    plt.show()

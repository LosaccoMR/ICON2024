import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import MultiPolygon

#Tendency of mental disease
def tendency_mental_diseases(df1):
# Seleziona le colonne rilevanti per i disturbi mentali e l'anno
    disorders = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders', 'Eating disorders']
    years = df1['Year'].unique()

# Calcola la prevalenza media di ogni disturbo per ciascun anno
    trend_data = df1.groupby('Year')[disorders].mean()

# Crea il grafico delle tendenze
    plt.figure(figsize=(14, 8))

    for disorder in disorders:
       plt.plot(trend_data.index, trend_data[disorder], label=disorder)

    plt.xlabel('Anno')
    plt.ylabel('Prevalenza media')
    plt.title('Trend delle malattie mentali nel mondo dal 1990 al 2019')
    plt.legend()
    plt.grid(True)
    plt.show()

#Tendenza (non media) di schizofrenia e depressione (che mostrano correlazione negativa)
def tendency_schizophrenia_depression(df1):
    # Seleziona gli anni specifici
    selected_years = [1990, 1995, 2000, 2005, 2010, 2019]
    df_filtered = df1[df1['Year'].isin(selected_years)]
    
    # Calcola la prevalenza media di schizofrenia e depressione per gli anni selezionati
    mean_values = df_filtered.groupby('Year')[['Schizophrenia disorders', 'Depressive disorders']].mean().reset_index()
    
    # Crea il grafico a barre comparativo
    bar_width = 0.35
    index = range(len(selected_years))

    plt.figure(figsize=(12, 6))

    bar1 = plt.bar(index, mean_values['Schizophrenia disorders'], bar_width, label='Schizophrenia disorders')
    bar2 = plt.bar([i + bar_width for i in index], mean_values['Depressive disorders'], bar_width, label='Depressive disorders')

    plt.xlabel('Anno')
    plt.ylabel('Prevalenza media')
    plt.title('Confronto dell\'andamento medio dei disturbi di Schizofrenia e Depressione')
    plt.xticks([i + bar_width / 2 for i in index], selected_years)
    plt.legend()
    plt.grid(True)
    plt.show()

# Funzione per visualizzare l'istogramma comparativo dell'andamento medio di disturbi alimentari e ansia negli anni specifici
def plot_comparative_bar_chart_eating_anxiety(df1):
    # Seleziona gli anni specifici
    selected_years = [1990, 1995, 2000, 2005, 2010, 2019]
    df_filtered = df1[df1['Year'].isin(selected_years)]
    
    # Calcola la prevalenza media di disturbi alimentari e ansia per gli anni selezionati
    mean_values = df_filtered.groupby('Year')[['Eating disorders', 'Anxiety disorders']].mean().reset_index()
    
    # Crea il grafico a barre comparativo
    bar_width = 0.35
    index = range(len(selected_years))

    plt.figure(figsize=(12, 6))

    bar1 = plt.bar(index, mean_values['Eating disorders'], bar_width, label='Eating disorders')
    bar2 = plt.bar([i + bar_width for i in index], mean_values['Anxiety disorders'], bar_width, label='Anxiety disorders')

    plt.xlabel('Year')
    plt.ylabel('Mean Prevalence')
    plt.title('Confronto dell\'andamento medio dei disturbi alimentari e ansia')
    plt.xticks([i + bar_width / 2 for i in index], selected_years)
    plt.legend()
    plt.grid(True)
    plt.show()

#Matrice di correlazione
def correlation_heatmap(df1):
    df1.rename(columns={
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizofrenia',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety disorders',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar disorders',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating disorders'
    }, inplace=True)
    df1_variables = df1[["Schizophrenia disorders", "Depressive disorders", "Anxiety disorders", "Bipolar disorders", "Eating disorders"]]
    Corrmat = df1_variables.corr()
    plt.figure(figsize=(10, 8), dpi=200)
    sns.heatmap(Corrmat, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matrice di Correlazione per le Malattie Mentali')
    plt.show()


# Funzione per calcolare la pendenza della prevalenza nel tempo per una nazione e un disturbo specifico

def calculate_trend(df1, country, disorder):
    country_data = df1[df1['Entity'] == country]
    slope, intercept = np.polyfit(country_data['Year'], country_data[disorder], 1)
    return slope

# Funzione per analizzare e visualizzare i trend dei disturbi mentali
def analyze_and_plot_trends(df1):
    disorders = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders', 'Eating disorders']
    trends_list = []

    for country in df1['Entity'].unique():
        for disorder in disorders:
            trend = calculate_trend(df1, country, disorder)
            trends_list.append({'Country': country, 'Disorder': disorder, 'Trend': trend})

    trend_data = pd.DataFrame(trends_list)
    increasing_trends = trend_data[trend_data['Trend'] > 0]

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    trend_geo = world.set_index('name').join(increasing_trends.set_index('Country'))
    trend_geo['geometry'] = trend_geo['geometry'].apply(lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x]))

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'choropleth'}]]
    )

    colors = {
        'Schizophrenia disorders': 'Blues',
        'Eating disorders': 'Oranges',
        'Bipolar disorders': 'Greens',
        'Depressive disorders': 'Reds',
        'Anxiety disorders': 'Purples'
    }

    for disorder, color in colors.items():
        disorder_data = trend_geo[trend_geo['Disorder'] == disorder]
        if not disorder_data.empty:
            fig.add_trace(
                go.Choropleth(
                    locations=disorder_data.index,
                    z=disorder_data['Trend'],
                    locationmode='country names',
                    colorscale=color,
                    showscale=False,
                    name=disorder
                )
            )

    # Definizione della legenda con colori esadecimali
    color_hex = {
        'Blues': '#1f77b4',
        'Oranges': '#ff7f0e',
        'Greens': '#2ca02c',
        'Reds': '#d62728',
        'Purples': '#9467bd'
    }

    legend_annotations = [
        dict(
            x=0.98,
            y=0.95 - (i * 0.05),
            xref='paper',
            yref='paper',
            showarrow=False,
            text=f"<b>{disorder}</b>",
            bgcolor=color_hex[color],
            opacity=0.8,
            bordercolor='black',
            borderwidth=1
        ) for i, (disorder, color) in enumerate(colors.items())
    ]

    fig.update_layout(
        title_text='Nazioni con incidenza di malattie mentali in crescita',
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        margin=dict(r=20, t=40, l=20, b=20),
        annotations=legend_annotations
    )

    fig.show()
#Subplot
def subplot_major_bipolar(df2):
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001)
    x1 = ["Andean Latin America", "West Sub-Saharan Africa", "Tropical Latin America", "Central Asia", "Central Europe",
          "Central Sub-Saharan Africa", "Southern Latin America", "North Africa/Middle East",
          "Southern Sub-Saharan Africa",
          "Southeast Asia", "Oceania", "Central Latin America", "Eastern Europe", "South Asia",
          "East Sub-Saharan Africa",
          "Western Europe", "World", "East Asia", "Caribbean", "Asia Pacific", "Australasia", "North America"]

    fig.append_trace(go.Bar(x=df2["Bipolar disorder"], y=x1, marker=dict(color='rgba(50, 171, 96, 0.6)',
                                                                         line=dict(color='rgba(20, 10, 56, 1.0)',
                                                                                   width=0)),
                            name='Bipolar disorder in Mental Health', orientation='h'), 1, 1)

    fig.append_trace(go.Scatter(x=df2["Major depression"], y=x1, mode='lines+markers', line_color='rgb(40, 0, 128)',
                                name='Major depression in Mental Health'), 1, 2)

    fig.update_layout(
        title='Major depression and Bipolar disorder',
        yaxis=dict(showgrid=False, showline=False, showticklabels=True, domain=[0, 0.85]),
        yaxis2=dict(showgrid=False, showline=True, showticklabels=False, linecolor='rgba(102, 102, 102, 0.8)',
                    linewidth=5, domain=[0, 0.85]),
        xaxis=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0, 0.45]),
        xaxis2=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0.47, 1], side='top',
                    dtick=10000),
        legend=dict(x=0.029, y=1.038, font_size=10),
        margin=dict(l=100, r=20, t=70, b=70),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        annotations=[dict(xref='x2', yref='y2', y=xd, x=ydn + 10, text='{:,}'.format(ydn) + '%',
                          font=dict(family='Arial', size=10, color='rgb(128, 0, 128)'), showarrow=False)
                     for ydn, yd, xd in zip(df2["Major depression"], df2["Bipolar disorder"], x1)] +
                    [dict(xref='x1', yref='y1', y=xd, x=yd + 10, text=str(yd) + '%',
                          font=dict(family='Arial', size=10, color='rgb(50, 171, 96)'), showarrow=False)
                     for ydn, yd, xd in zip(df2["Major depression"], df2["Bipolar disorder"], x1)] +
                    [dict(xref='paper', yref='paper', x=-0.2, y=-0.109, text="Visualizzazione della salute mentale",
                          font=dict(family='Arial', size=20, color='rgb(150,150,150)'), showarrow=False)]
    )
    fig.show()

#Line chart
def line_chart_depressive_symptoms(df3):
    x = ["Appetite change", "Average across symptoms", "Depressed mood", "Difficulty concentrating", "Loss of interest",
         "Low energy", "Low self-esteem", "Psychomotor agitation", "Psychomotor agitation", "Sleep problems", "Suicidal ideation"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df3["Nearly every day"], name='Nearly every day', line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=x, y=df3["More than half the days"], name='More than half the days', line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=x, y=df3["Several days"], name='Several days', line=dict(color='black', width=4, dash='dashdot')))
    fig.update_layout(title='Depressive symptoms across us population', xaxis_title='Entity', yaxis_title='Types of days')
    fig.show()

def line_chart_mental_illness(df4):
    x = ["Alcohol use disorders", "Amphetamine use disorders", "Anorexia nervosa", "Anxiety disorders",
         "Attention-deficit hyperactivity disorder", "Autism spectrum disorders", "Bipolar disorder",
         "Bulimia nervosa", "Cannabis use disorders", "Cocaine use disorders", "Dysthymia","Major depressive disorder",
         "Opioid use disorders", "Other drug use disorders", "Personality disorders"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df4["Number of countries with primary data on prevalence of mental disorders"], name='Nearly every day', line=dict(color='firebrick', width=4)))
    fig.update_layout(title='Malattie mentali nello studio del carico globale di malattia', xaxis_title='Malattie', yaxis_title='Numero di paesi')
    fig.show()

#Box plot
def box_plots(df1):
    df1.rename(columns={
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia disorders',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety disorders',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar disorders',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating disorders'
    }, inplace=True)
    df1_variables = df1[["Schizophrenia disorders", "Depressive disorders", "Anxiety disorders", "Bipolar disorders", "Eating disorders"]]
    Numerical = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders', 'Eating disorders']
    fig = make_subplots(rows=1, cols=5, subplot_titles=Numerical)
    for i in range(5):
        trace = go.Box(x=df1_variables[Numerical[i]], name=Numerical[i])
        fig.add_trace(trace, row=1, col=i+1)
    fig.update_layout(height=300, width=1200, title_text="Boxplots")
    fig.update_layout(showlegend=False)
    fig.show()


import os
import pandas as pd
import data_visualization as dv
import optimize_rf as opt

# Determina il percorso del file corrente
base_dir = os.path.dirname(os.path.abspath(__file__))

# Lettura dei dati da file CSV
Data1 = pd.read_csv(os.path.join(base_dir, '..', 'DbDefinitivi', 'DisturbiMentali-DalysNazioniDelMondo.csv'))
Data2 = pd.read_csv(os.path.join(base_dir, '..', 'DbDefinitivi', '4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv'))
Data3 = pd.read_csv(os.path.join(base_dir, '..', 'DbOriginali', '6- depressive-symptoms-across-us-population.csv'))
Data4 = pd.read_csv(os.path.join(base_dir, '..', 'DbDefinitivi', '7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv'))

# Creazione dei DataFrame
df1 = pd.DataFrame(Data1)
df2 = pd.DataFrame(Data2)
df3 = pd.DataFrame(Data3)
df4 = pd.DataFrame(Data4)

# Filtrare i dati per l'Italia
data_italy = Data1[Data1['Entity'] == 'Italy']

# Elenco delle patologie e dei rispettivi DALYs
dalys_columns = [
    'DALYs Cause: Depressive disorders', 'DALYs Cause: Schizophrenia',
    'DALYs Cause: Bipolar disorder', 'DALYs Cause: Eating disorders',
    'DALYs Cause: Anxiety disorders'
]

# Elenco di tutte le metriche presenti nel dataset
all_columns = Data1.columns.tolist()
# Rimuovere le colonne che non sono metriche (Entity, Code, Year)
feature_columns = [col for col in all_columns if col not in ['Entity', 'Code', 'Year'] + dalys_columns]

# Aggiungere l'anno come feature
feature_columns.append('Year')

def main_menu():
    while True:
        print("\nMenu Principale:")
        print("1. Analisi Descrittiva del Dataset")
        print("2. Elaborazione dei Dati")
        print("0. Esci")

        main_choice = input("Inserisci la tua scelta: ")

        if main_choice == "1":
            while True:
                print("\nAnalisi Descrittiva del Dataset:")
                print("1. Grafico per visualizzare la tendenza media delle malattie mentali")
                print("2. Grafico a barre: Schizofrenia e Depressione")
                print("3. Grafico a barre: Ansia e Disturbi alimentari")
                print("4. Matrice di Correlazione")
                print("5. Nazioni con incremento di malattie mentali")
                print("6. Subplot: Depressione Maggiore e Disturbo Bipolare")
                print("7. Grafico a Linee: Sintomi Depressivi")
                print("8. Grafico a Linee: Malattie Mentali")
                print("9. Box Plot")
                print("0. Torna al Menu Principale")

                choice = input("Inserisci la tua scelta: ")

                if choice == "1":
                    dv.tendency_mental_diseases(df1)
                elif choice == "2":
                    dv.tendency_schizophrenia_depression(df1)
                elif choice == "3":
                    dv.plot_comparative_bar_chart_eating_anxiety(df1)
                elif choice == "4":
                    dv.correlation_heatmap(df1)
                elif choice == "5":
                    dv.analyze_and_plot_trends(df1)
                elif choice == "6":
                    dv.subplot_major_bipolar(df2)
                elif choice == "7":
                    dv.line_chart_depressive_symptoms(df3)
                elif choice == "8":
                    dv.line_chart_mental_illness(df4)
                elif choice == "9":
                    dv.box_plots(df1)
                elif choice == "0":
                    break
                else:
                    print("Scelta non valida. Per favore, riprova.")

        elif main_choice == "2":
            while True:
                print("\nElaborazione dei Dati:")
                print("1. Previsione malattie mentali in Italia")
                print("0. Torna al Menu Principale")

                choice = input("Inserisci la tua scelta: ")

                if choice == "1":
                        opt.predict_dalys(feature_columns, dalys_columns, data_italy)
                elif choice == "0":
                    break
                else:
                    print("Scelta non valida. Per favore, riprova.")

        elif main_choice == "0":
            break
        else:
            print("Scelta non valida. Per favore, riprova.")


if __name__ == "__main__":
    main_menu()
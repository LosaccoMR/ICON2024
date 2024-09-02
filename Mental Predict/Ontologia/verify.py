from rdflib import Graph, Namespace, RDF, URIRef
from rdflib.namespace import OWL

FUTURA = Namespace("http://futuramente.org/ontologies/2023#")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# percorso per l'ontologia generata
generated_ontology_path = 'Europa/Risultati/IntegratedOntology.owl'

# Carica l'ontologia generata
g_generated = Graph()
g_generated.parse(generated_ontology_path)

print(f"Numero di tripli nell'ontologia generata: {len(g_generated)}")

# Visualizza le classi nell'ontologia
for s in g_generated.subjects(RDF.type, OWL.Class):
    print(f"Classe: {s}")

# Visualizza le proprietà nell'ontologia
for s in g_generated.subjects(RDF.type, OWL.ObjectProperty):
    print(f"Proprietà: {s}")

# Visualizza gli individui (nazioni) e le loro proprietà
for s in g_generated.subjects(RDF.type, FUTURA.Country):
    print(f"Individuo: {s}")
    for p, o in g_generated.predicate_objects(subject=s):
        print(f"  Proprietà: {p}, Valore: {o}")

# Mappatura dei disturbi ai loro URI
disorder_uris = {
    "Schizophrenia": "http://purl.obolibrary.org/obo/DOID_5419",
    "Depressive": "http://purl.obolibrary.org/obo/DOID_1596",
    "Anxiety": "http://purl.obolibrary.org/obo/DOID_2030",
    "Bipolar": "http://purl.obolibrary.org/obo/DOID_3312",
    "Eating": "http://purl.obolibrary.org/obo/DOID_8670"
}

# Verifica che le relazioni siano corrette
for country in g_generated.subjects(RDF.type, FUTURA.Country):
    for disorder_label, disorder_uri in disorder_uris.items():
        if (country, FUTURA.hasDisorder, URIRef(disorder_uri)) in g_generated:
            print(f"{country} ha il disturbo {disorder_label}")
        else:
            print(f"ERRORE: {country} non ha il disturbo {disorder_label}")

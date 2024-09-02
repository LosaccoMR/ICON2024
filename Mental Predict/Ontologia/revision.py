from rdflib import Graph, Namespace, RDF, URIRef
from rdflib.namespace import OWL

# Definire i namespace
FUTURA = Namespace("http://futuramente.org/ontologies/2023#")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# Caricare l'ontologia generata
generated_ontology_path = 'Europa/Risultati/IntegratedOntology.owl'
g_generated = Graph()
g_generated.parse(generated_ontology_path)

# Funzione per verificare se una URI appartiene a un namespace noto
def is_known_namespace(uri):
    known_namespaces = [FUTURA, OBO, OWL]
    return any(str(uri).startswith(str(ns)) for ns in known_namespaces)

# Verificare tutte le classi
print("Classi non standard:")
for s in g_generated.subjects(RDF.type, OWL.Class):
    if not is_known_namespace(s):
        print(f"Classe: {s}")

# Verificare tutte le proprietà
print("\nProprietà non standard:")
for p in g_generated.predicates():
    if not is_known_namespace(p):
        print(f"Proprietà: {p}")

# Verificare gli individui e le loro proprietà
print("\nIndividui e loro proprietà:")
for s in g_generated.subjects(RDF.type, FUTURA.Country):
    print(f"Individuo: {s}")
    for p, o in g_generated.predicate_objects(subject=s):
        if not is_known_namespace(p):
            print(f"  Proprietà non standard: {p}, Valore: {o}")
        else:
            print(f"  Proprietà: {p}, Valore: {o}")



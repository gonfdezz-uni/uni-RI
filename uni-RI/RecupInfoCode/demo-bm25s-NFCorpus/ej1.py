import bm25s
import Stemmer

#Set para los documentos relevantes de mi consulta: PLAIN-382
relevant_docs_plain382 = set()

#Cargo los documentos indicados como relevantes para las consulta PLAIN-382
try:
    with open("qrels.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                id_consultation = parts[0]
                id_relevant_doc = parts[2]
                if id_consultation == "PLAIN-382":
                    relevant_docs_plain382.add(id_relevant_doc)
                    
    print(f"Cargados {len(relevant_docs_plain382)} documentos relevantes oficiales para PLAIN-382.")
    print(relevant_docs_plain382)

except FileNotFoundError:
    print("Error: No se encuentra el archivo qrels.txt")
#Cargo los documentos indicados como relevantes para las consulta PLAIN-382

retriever = bm25s.BM25.load("pubmed", load_corpus=True)

while True:
    try:
        query = input("Enter your search query: ").strip()
        if query:
            break
        print("Query cannot be empty. Please try again.")
    except (KeyboardInterrupt, EOFError):
        query = None

stemmer = Stemmer.Stemmer("english")
query_tokenized = bm25s.tokenize(query, stemmer=stemmer, stopwords="en", lower=True)

# RECORDAR AÑADIR A LA DOCUMENTACIÓN LA CHARLA CON EL CHATBOT

# Pido 100 para así hacer el proceso de eliminación de duplicados
results, scores = retriever.retrieve(
    query_tokenized, k=100, corpus=retriever.corpus, show_progress=False
)

# Código nuevo para la eliminación de duplicados
documentos_unicos = []
vistos = set()

for i in range(results.shape[1]):
    result = results[0, i]
    score = scores[0, i]
    # Esto sirve para agrupar por título y puntuación redondeada
    clave = (result["title"], round(score, 4))
    if clave not in vistos:
        vistos.add(clave)
        documentos_unicos.append(result)
    if len(documentos_unicos) == 10:
        break
# Ahora 'documentos_unicos' tiene los 10 primeros documentos distintos para mi análisis
print("\nTop 10 documentos únicos (sin duplicados):\n")
for rank, result in enumerate(documentos_unicos):
    print(f"{rank + 1}. {result['id']}\t{result['title']}")

#---Cálculo de la P@10---

recuperados_relevantes = 0
total_recuperados = len(documentos_unicos) #10


print(f"--- Calculando precisión para {total_recuperados} documentos ---")

for linea in documentos_unicos:
    doc_id = linea['id']
    if doc_id in relevant_docs_plain382:
        recuperados_relevantes += 1

#Calcular precisión
if total_recuperados > 0:
    precision = recuperados_relevantes / total_recuperados

print(f"\nPrecisión: {precision:.4f} ({recuperados_relevantes}/{total_recuperados})")


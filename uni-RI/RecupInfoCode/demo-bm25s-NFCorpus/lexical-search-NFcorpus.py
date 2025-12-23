import bm25s
import Stemmer

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

# --PARA PREGUNTAR EN CLASE--
# ¿Hay que valorar la relevancia por código o en papel?
# --PARA PREGUNTAR EN CLASE--

print("\nTop 10 documentos únicos (sin duplicados):\n")
for rank, result in enumerate(documentos_unicos):
    print(f"{rank + 1}. {result['id']}\t{result['title']}")

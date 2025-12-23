import bm25s
import Stemmer

retriever = bm25s.BM25.load("pubmed", load_corpus=True)
stemmer = Stemmer.Stemmer("english")

# RECORDAR AÑADIR A LA DOCUMENTACIÓN LA CHARLA CON EL CHATBOT

# Lee las consultas del archivo
consultas = []
contador = 0
with open("NFcorpus-questions-selection.txt") as f:
    for linea in f:
        if linea.startswith("PLAIN-"):
            contador+=1
        # Divido las consultas en 3 partes: id, url y el texto
        partes = linea.strip().split("\t", 2)
        consultas.append({"id": partes[0], "url": partes[1], "text": partes[2]})

# --MÉTRICAS--
metricas_por_consulta = []

for consulta in consultas:
    # Saco las preguntas de cada consulta
    query_text = consulta["text"]
    query_tokenized = bm25s.tokenize(
        query_text, stemmer=stemmer, stopwords="en", lower=True
    )
    # Pido tantas como consultas PLAIN tiene el documento
    results, scores = retriever.retrieve(
        query_tokenized, k=contador, corpus=retriever.corpus, show_progress=False
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
        if len(documentos_unicos) == 100:
            break


# --PARA PREGUNTAR EN CLASE--
# ¿Hay que valorar la relevancia por código o en papel?
# --PARA PREGUNTAR EN CLASE--

print("\nTop 100 documentos únicos (sin duplicados) (hacer relevancia):\n")
for rank, doc in enumerate(documentos_unicos):
    print(f"{rank + 1}. {doc['id']}\tScore:{scores[0, rank]:.4f}\t{doc['title']}")

# --- PAUSA: aquí debes marcar a mano cuántos son relevantes
    # Puedes copiar la lista, anotar los índices relevantes, y calcular las métricas por consulta
    # Guarda el resultado (opcional)
    # input("Presiona ENTER para continuar con la siguiente consulta...")

print("\n--- Fin del proceso automático. Marca relevancia y calcula métricas a mano ---")


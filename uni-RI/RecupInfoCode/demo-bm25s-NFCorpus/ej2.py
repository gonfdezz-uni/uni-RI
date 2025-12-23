import bm25s
import Stemmer

# --- CONFIGURACIÓN DE ARCHIVOS ---
ARCHIVO_CONSULTAS = "NFcorpus-questions-selection.txt"
ARCHIVO_QRELS = "qrels.txt"
TOP_K = 100  # P@100, R@100

# 1. CARGAR QRELS
# Diccionario de conjuntos para cada consulta con sus documentos relevantes
qrels = {}

try:
    with open(ARCHIVO_QRELS, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                id_consultation = parts[0]
                id_relevant_doc= parts[2]
                # Inicializamos el set para este ID si no existe
                if id_consultation not in qrels:
                    qrels[id_consultation] = set()
                qrels[id_consultation].add(id_relevant_doc)
except FileNotFoundError:
    print(f"ERROR: No se encontró el archivo {ARCHIVO_QRELS}")
    exit()

retriever = bm25s.BM25.load("pubmed", load_corpus=True)
stemmer = Stemmer.Stemmer("english")

#Listas para las métricas sobre las que luego haré promedio
all_precisions = []
all_recalls = []
all_f1s = []

# 3. PROCESAR EL ARCHIVO DE CONSULTAS
try:
    with open(ARCHIVO_CONSULTAS, "r") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line: 
                continue
            
            # PARSEO: ID  URL  TEXTO
            # Usamos maxsplit=2 para separar solo los dos primeros bloques de espacio/tabs
            # parts[0] = ID, parts[1] = URL, parts[2] = TEXTO COMPLETO
            parts = line.split(maxsplit=2)
            
            if len(parts) < 3:
                continue # Saltamos líneas vacías o mal formadas
                
            id_consultation = parts[0]       # Ej: PLAIN-2
            # url = parts[1]     # La URL la ignoramos
            query_text = parts[2] # Ej: "Do Cholesterol Statin Drugs..."
            
            # --- A. BÚSQUEDA ---
            query_tokenized = bm25s.tokenize(query_text, stemmer=stemmer, stopwords="en", lower=True)
            
            # Pedimos 200 para tener margen de sobra al eliminar duplicados
            results, scores = retriever.retrieve(query_tokenized, k=200, corpus=retriever.corpus, show_progress=False)
            
            # --- B. ELIMINACIÓN DE DUPLICADOS ---
            documentos_unicos = []
            vistos = set()
            
            # Iteramos sobre los resultados devueltos (bm25s devuelve matriz [1, k])
            for i in range(results.shape[1]):
                res = results[0, i]
                score = scores[0, i]
                
                # Tu clave de unicidad: Título + Score redondeado
                clave = (res["title"], round(score, 4))
                
                if clave not in vistos:
                    vistos.add(clave)
                    documentos_unicos.append(res)
                
                # Paramos exactamente al llegar a TOP_K (100)
                if len(documentos_unicos) == TOP_K:
                    break
            
            # --- C. CÁLCULO DE MÉTRICAS ---
            
            # Obtenemos el conjunto de documentos 'correctos' para esta consulta
            relevantes_esperados = qrels.get(id_consultation, set())
            total_relevantes_existentes = len(relevantes_esperados)
            
            # Si la consulta no está en el qrels, no podemos evaluarla -> saltamos
            if total_relevantes_existentes == 0:
                continue
                
            # Contamos cuántos de nuestros 100 recuperados son relevantes
            aciertos = 0
            for doc in documentos_unicos:
                doc_id = doc.get("_id") or doc.get("id")
                if doc_id in relevantes_esperados:
                    aciertos += 1
            
            # 1. Precisión @ 100
            # (Aciertos / Total documentos mostrados)
            p_val = aciertos / TOP_K
            
            # 2. Exhaustividad (Recall) @ 100
            # (Aciertos / Total relevantes que existen en la base de datos)
            r_val = aciertos / total_relevantes_existentes
            
            # 3. F1 Score @ 100
            # Fórmula estándar: 2 * (P*R) / (P+R)
            # NOTA: Si te piden estrictamente (P*R)/(P+R), borra el "2 *" del principio.
            if (p_val + r_val) > 0:
                f1_val = 2 * (p_val * r_val) / (p_val + r_val)
            else:
                f1_val = 0.0
                
            # Guardamos en las listas generales
            all_precisions.append(p_val)
            all_recalls.append(r_val)
            all_f1s.append(f1_val)
            
            count += 1

except FileNotFoundError:
    print(f"ERROR: No se encontró {ARCHIVO_CONSULTAS}")
    exit()

# 4. RESULTADOS PROMEDIO
print("\n" + "="*50)
print(f"RESULTADOS GLOBALES (Promedio sobre {len(all_precisions)} consultas)")
print("="*50)

if len(all_precisions) > 0:
    avg_p = sum(all_precisions) / len(all_precisions)
    avg_r = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print(f"Precisión Media (P@100):      {avg_p:.4f}")
    print(f"Exhaustividad Media (R@100):  {avg_r:.4f}")
    print(f"F1-Score Medio (@100):        {avg_f1:.4f}")
else:
    print("No se generaron métricas. Verifica que los IDs de 'queries' coincidan con 'qrels'.")



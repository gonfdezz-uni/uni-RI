import bm25s
import Stemmer

corpus_verbatim = []
corpus_plaintext = []

with open("doc_dump.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 4:
            continue

        doc_id = parts[0]
        title = parts[2]
        abstract = parts[3]
        doc = {"id": doc_id, "title": title, "text": abstract}
        corpus_verbatim.append(doc)
        corpus_plaintext.append(f"{title} {abstract}")

print(f"Found {len(corpus_verbatim)} documents to index.")

stemmer = Stemmer.Stemmer("english")

corpus_tokenized = bm25s.tokenize(
    corpus_plaintext, stopwords="en", stemmer=stemmer, lower=True, show_progress=True
)

retriever = bm25s.BM25(method="lucene", idf_method="lucene")
retriever.index(corpus_tokenized, show_progress=True)

retriever.save("pubmed", corpus=corpus_verbatim)

print(f"Index saved.")  # noqa: F541

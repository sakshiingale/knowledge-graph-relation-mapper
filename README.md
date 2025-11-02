# 🧠 Knowledge Graph Relation Mapper

> **An NLP-powered knowledge graph for intelligent relation mapping.**
> Visualizing semantic relationships between text sections using embeddings and graph theory.
**Knowledge Graph Relation Mapper** builds an **interactive graph** that maps conceptual relationships between text sections or entities using **natural language processing (NLP)**.
It generates **Sentence Transformer (BERT-based)** embeddings and uses **cosine similarity** to detect related content, visualizing it as a connected knowledge network.

---

## 🧰 Technologies Used

| Category          | Tools / Libraries                   |
| ----------------- | ----------------------------------- |
| Language          | Python                              |
| NLP               | Sentence Transformers (BERT), spaCy |
| Graph             | NetworkX, PyVis                     |
| Similarity Metric | Cosine Similarity                   |
| Visualization     | PyVis, Matplotlib                   |

---


## 🔢 How It Works

1. Convert text sections into embeddings using **Sentence Transformers**
2. Compute **cosine similarity** between all pairs
3. Apply a threshold (default: 0.6) to define connections
4. Build a graph using **NetworkX**
5. Visualize the result interactively with **PyVis** or store in **Neo4j**


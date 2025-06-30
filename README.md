# Deep Learning-Powered Literature Review: Automating Knowledge Synthesis with E5 Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/priyabhayani19/DL-Smarter-Literature-Review-with-Embedding-/blob/main/Deep_Learning_Powered_Literature_Review_Automating_Knowledge_Synthesis.ipynb)

## Overview

This project presents an end-to-end automated literature review pipeline that leverages state-of-the-art transformer-based sentence encoders—specifically the **E5 embedding model**—to deliver truly semantic embeddings for academic paper abstracts. Unlike traditional vector-space models like TF-IDF, the E5 embedding model captures deep semantic relationships, improving the relevance and coherence of clustered topics.

The system automates:

- Harvesting metadata and abstracts from multiple academic databases.
- Cleaning and deduplicating collected text data.
- Computing high-dimensional semantic embeddings using the E5 model.
- Filtering papers using a cosine similarity threshold against a set of model case papers.
- Clustering the filtered papers using MiniBatch-KMeans with t-SNE visualization to reveal thematic structures.

Applied to a corpus of around 1,000 candidate papers, this approach distilled about 500 highly relevant papers, grouped into coherent clusters within an hour, demonstrating efficiency and semantic accuracy.

For detailed methodology and results, see **Report1.pdf**.

The `.ipynb` file is designed to run seamlessly on Google Colab.

---

## Steps for Usage of the E5 Embedding Literature Review Pipeline

### 1. Data Collection using Findpapers

**Setup:**

- Install the Findpapers library ([Findpapers GitHub](https://github.com/jonatasgrosman/findpapers)) in your Colab environment.

**Configure Search:**

- Define detailed search queries with relevant keywords and logical operators (AND, OR).
- Set publication date range (e.g., `SINCE = '2013-01-01'`, `UNTIL = '2025-05-01'`).
- Specify the number of papers to retrieve (e.g., `NUM = 1000`).
- Choose databases such as PubMed, arXiv, and Scopus.
- Set necessary API keys (e.g., `SCOPUS_API_KEY`) as environment variables.

**Run Search:**

- Execute Findpapers to retrieve metadata and abstracts.
- Output is saved in a JSON file containing detailed paper information.

---

### 2. Exporting Data from JSON to Excel

- Load the JSON file using Python’s `pandas`.
- Extract key metadata: titles, publication dates, abstracts, authors, databases, publishers, journal names, keywords, DOIs, citation counts.
- Compile data into a DataFrame and export as an Excel file for easier processing.

---

### 3. Cleaning Abstract Text

- Clean abstracts by removing HTML tags, handling missing or non-English entries, and eliminating unnecessary symbols.
- Normalize all text to lowercase to maintain consistency for embedding.

---

### 4. Generating Semantic Embeddings with E5

- Use the `sentence_transformers` library to embed cleaned abstracts with the **E5-base (intfloat/e5-base) and All-MiniLM-L6-v2 models**.
- Generate dense, high-dimensional semantic vectors that capture nuanced textual meaning beyond keyword matching.

---

### 5. Filtering and Clustering Semantic Embeddings

**Cosine Similarity Filtering:**

- A user-defined **query sentence** is embedded using the same model.
- Compute cosine similarity between query and each paper abstract.
- Keep only papers with similarity score ≥ **0.80**.
- Filtered dataset is saved to `most_similar_paperse5.xlsx`.

**Determining Number of Clusters (Elbow Method):**

- Compute Sum of Squared Errors (SSE) for different cluster counts.
- Identify optimal cluster number by finding the "elbow" in the SSE plot.

**Clustering with MiniBatchKMeans:**

- Cluster the filtered embedding matrix using MiniBatchKMeans for scalable performance.
- Group semantically similar papers into coherent thematic clusters.

**Cluster Visualization using t-SNE**
- High-dimensional embeddings are reduced using:
  - **PCA** → reduces dimensions to 20 for performance.
  - **t-SNE** → maps embeddings to 2D space for visualization.
- Clusters are plotted as scatter plots, colored by cluster assignment, for easy interpretation.

### 6. Cluster Analysis & Summarization
- Extract and rank **top 10 keywords** per cluster for quick topic labeling.
- Use **T5-small** summarization to generate a **short summary** of each cluster’s content.
- Visual tools (bar/pie charts) show keyword and paper distribution across clusters.

### 7. Trend Analysis & Source Evaluation
- Visualize publication trends (year-wise growth).
- Analyze paper distribution by source database (PubMed, arXiv, Scopus).
- Identify which clusters contribute most toward specific themes (e.g., VR-based therapy, mental health).

  ## Performance Metrics (Evaluation Summary)

| Metric         | E5-Base | All-MiniLM | TF-IDF |
|----------------|---------|------------|--------|
| **P@10**       | 0.90    | 0.80       | 0.70   |
| **MRR**        | 1.00    | 1.00       | 0.33   |
| **nDCG@10**    | 0.93    | 0.85       | 0.70   |
| **Silhouette** | 0.0258  | 0.0221     | 0.0105 |
| **DB Index**   | 4.41    | 3.49       | 6.80   |

- **E5**: Best for top-ranked relevance and early precision.
- **All-MiniLM**: Slightly better clustering compactness.
- **TF-IDF**: Weakest across both retrieval and cluster coherence.



**Saving Results:**

-`most_similar_paperse5.xlsx`: Top filtered papers based on semantic similarity.
- `clustered_papers.xlsx`: Final dataset with cluster assignments.
- `plots/`: Visualizations for SSE, t-SNE, keyword frequency, publication trends, and database source contribution.


## Files Included

- `E5_Embedding_Literature_Review.ipynb` — The main Colab notebook for running the pipeline.
- `Report1.pdf` — Detailed project report describing methodology, experiments, and results.
- Additional data and output files generated during execution.

---

## Requirements

- Python 3.x
- Libraries: `pandas`, `sentence_transformers`, `scikit-learn`, `findpapers`, `numpy`, `matplotlib`, `openpyxl`, `TenserFlow`
- Access to APIs for academic databases (as needed)




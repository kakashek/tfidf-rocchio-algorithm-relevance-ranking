# TFIDF Rocchio Algorithm Relevance Feedback Model
This project implements a vector space information retrieval (IR) model that combines **TF-IDF weighting** with the **Rocchio relevance feedback algorithm** to improve document ranking accuracy for a given query.

## Overview
The model performs ranked document retrieval by representing queries and documents as **term-weighted vectors**. Ranking is done based on the similarity (dot product) between document vectors and an expanded query vector, refined using pseudo-relevance feedback.

## Steps

### 1. **TF-IDF Vectorization**
Each document is converted into a vector using the **Term Frequency-Inverse Document Frequency** (TF-IDF) scheme:
- **TF**: Measures how frequently a term appears in a document.
- **IDF**: Measures how unique a term is across all documents.
- Both are log-scaled and normalized using the L2 norm to prevent bias from longer documents.

### 2. **Query Parsing and Expansion**
Queries are extracted from `Queries.txt` and have different weights:
- `<title>` terms: 1.5
- `<desc>` terms: 0.75
- **WordNet synonyms** for title terms: 0.5

### 3. **Initial Ranking**
The system calculates the similarity between the query vector and each document's TF-IDF vector using the **dot product** (since all vectors are normalized, this acts like cosine similarity).

### 4. **Rocchio Relevance Feedback**
- Top 10 ranked documents are assumed to be **relevant** and remaining documents are treated as **non-relevant**.
- A new adjusted query vector is generated using the Rocchio formula
- Parameters used:
  - α = 1 (original query weight)
  - β = 2 (boost for relevant documents)
  - γ = 0.5 (penalty for non-relevant documents)

## Evaluation Metrics

The model is evaluated using standard IR performance metrics:

1. **Mean Average Precision (MAP)**  
   Measures precision across all relevant document positions.

2. **Precision@12**  
   Measures how many relevant documents appear in the top 12 results.

3. **DCG@12 (Discounted Cumulative Gain)**  
   Captures the ranking quality by assigning higher weights to relevant documents appearing earlier in the list.

Results are generated and written to `Output/` for each dataset, including both individual and average metrics.




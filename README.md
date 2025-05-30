# Run performance and semantic quality benchmarks on OpenAI and Cohere embedding models.

# Sentence Embedding Benchmark with Cohere and OpenAI

This project demonstrates how to generate sentence embeddings using two popular NLP APIs — Cohere and OpenAI — and compares their embedding generation times and cosine similarity scores between pairs of sentences.

---

## Features

- Generate embeddings using **Cohere's** embedding API (`embed-english-v3.0` model).
- Generate embeddings using **OpenAI's** embedding API (`text-embedding-ada-002` model).
- Calculate and compare cosine similarity between sentence pairs.
- Measure and display the time taken for embedding generation.
- Simple error handling for API requests.

---

## Requirements

- Python 3.7+
- `openai` Python SDK
- `cohere` Python SDK
- `numpy`
- `scikit-learn`

You can install the dependencies with:

```bash
pip install openai cohere numpy scikit-learn

##Sample Output

import time
import openai
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Replace these with your actual API keys ---
openai.api_key = "OPENAI KEY"
co = cohere.Client("lE3LZMlWewIPD8fSdGHKQgKrGJwektk2zJzT67wj")

sentences1 = ["A man is eating food.", "A child is playing with a dog."]
sentences2 = ["A man is eating pasta.", "A kid is playing with a puppy."]


def encode_cohere(texts, model="embed-english-v3.0"):
    try:
        start = time.time()
        response = co.embed(
            texts=texts,
            model=model,
            input_type="search_document"  
        )
        embeddings = [np.array(embedding) for embedding in response.embeddings]
        elapsed = time.time() - start
        return embeddings, elapsed
    except Exception as e:
        print("Cohere API error:", e)
        return None, None


def encode_openai(texts, model="text-embedding-ada-002"):
    try:
        start = time.time()
        response = openai.embeddings.create(
            model=model,
            input=texts
        )
        embeddings = [np.array(item.embedding) for item in response.data]
        elapsed = time.time() - start
        return embeddings, elapsed
    except Exception as e:
        print("OpenAI API error:", e)
        return None, None


def benchmark(name, encoder):
    print(f"\nRunning {name} benchmark...")

    embs1, time1 = encoder(sentences1)
    if embs1 is None:
        print(f"{name}: Failed to generate embeddings for sentences1.")
        return

    embs2, time2 = encoder(sentences2)
    if embs2 is None:
        print(f"{name}: Failed to generate embeddings for sentences2.")
        return

    print(f"{name} embedding time for sentences1: {time1:.3f} seconds")
    print(f"{name} embedding time for sentences2: {time2:.3f} seconds")
    print(" Embeddings generated")

    for i, (e1, e2) in enumerate(zip(embs1, embs2)):
        score = cosine_similarity([e1], [e2])[0][0]
        print(f"Pair {i+1}: Cosine Similarity = {score:.4f}")


benchmark("Cohere", encode_cohere)
benchmark("OpenAI", encode_openai)

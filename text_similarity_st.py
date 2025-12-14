from sentence_transformers import SentenceTransformer, util

# Load a pre-trained model (e.g., all-MiniLM-L6-v2)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the texts to compare
sentence1 = "The cat sat on the mat."
sentence2 = "A feline rested on a rug."
sentence3 = "The car is driving down the street."

# Compute embeddings
embedding_1 = model.encode(sentence1, convert_to_tensor=True)
embedding_2 = model.encode(sentence2, convert_to_tensor=True)
embedding_3 = model.encode(sentence3, convert_to_tensor=True)

# Compute cosine similarity
# The score ranges from -1 to 1, where 1 means identical
similarity_1_2 = util.pytorch_cos_sim(embedding_1, embedding_2).item()
similarity_1_3 = util.pytorch_cos_sim(embedding_1, embedding_3).item()

print(f"Similarity between '{sentence1}' and '{sentence2}': {similarity_1_2:.4f}")
print(f"Similarity between '{sentence1}' and '{sentence3}': {similarity_1_3:.4f}")

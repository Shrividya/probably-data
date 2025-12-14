from sentence_transformers import SentenceTransformer

# Load a pre-trained model
# all-MiniLM-L6-v2 is a popular, fast, and effective general-purpose model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample text data (a list of sentences)
sentences = [
    "Vector embeddings are a powerful tool in machine learning.",
    "They convert text into a dense numerical representation.",
    "Similar sentences will have vectors that are close together in the vector space."
]

# Encode the sentences to get the embeddings
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding snippet: {embedding[:5]}...\n") # Print first 5 dimensions

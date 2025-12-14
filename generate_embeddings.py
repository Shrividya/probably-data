import os
from openai import OpenAI

# Initialize the OpenAI client (API key is automatically picked up from env variable)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_openai_embedding(text, model="text-embedding-3-small"):
    """Converts a single string into a vector embedding using the OpenAI API."""
    # The text needs to be formatted for the API
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Example usage
text_to_embed = "Sample code for creating vector embeddings using the OpenAI API."
embedding = get_openai_embedding(text_to_embed)

print(f"Text: {text_to_embed}")
print(f"Embedding length: {len(embedding)}") # The default length is 1536
print(f"Embedding snippet: {embedding[:5]}...")

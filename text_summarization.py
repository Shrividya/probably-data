from transformers import pipeline

# Load a pre-trained summarization model
# 'sshleifer/distilbart-cnn-12-6' is a good, fast choice
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Sample text to summarize
long_text = """
We start our day with technology: our phone alarms wake us up, and we use coffee makers or smart appliances in the kitchen. In education, physical textbooks are often replaced by tablets or laptops, making learning more interactive with online videos and digital courses. Communication has become incredibly fast and simple. Instead of sending physical letters and waiting days for a reply, we can instantly connect with anyone around the world using video calls, social media, and instant messaging apps. 
Medical technology has also seen remarkable progress. Advanced machines and devices allow doctors to perform complex operations and diagnose illnesses more accurately, which has significantly increased the average lifespan. 
However, this reliance on technology comes with challenges. Issues like data privacy, cyberbullying, and job displacement due to automation are real concerns. Overuse of gadgets can also lead to health problems and social isolation, reminding us that we need to balance screen time with real-life interactions and physical activity. Ultimately, technology itself is a powerful tool, and its impact—whether positive or negative—depends on how we choose to use it
"""

# Generate the summary
# Adjust max_length and min_length as needed for different text lengths
summary = summarizer(long_text, max_length=60, min_length=30, do_sample=False)

print("Original Text:")
print(long_text)
print("\nGenerated Summary:")
print(summary[0]['summary_text'])

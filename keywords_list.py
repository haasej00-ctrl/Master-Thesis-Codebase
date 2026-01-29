import pickle

keywords_list = [
    "artificial intelligence",
    "machine learning",
    "business intelligence",
    "big data",
    "data science",
    "AI/ML",
    "AI",
    "ML",
    "data mining",
    "data scientist",
    "deep learning",
    "computer vision",
    "natural language processing",
    "chatbot",
    "image recognition",
    "object recognition",
    "large language model",
    "LLM",
    "machine translation",
    "support vector machine",
    "classification algorithm",
    "generative ai",
    "A.I.",
    "neural network",
    "supervised learning",
    "computational linguistic",
    "clustering algorithms",
    "recommender system",
    "dimensionality reduction",
    "information extraction",
    "kernel method",
    "unsupervised learning",
    "AI architecture",
    "transformer",
    "chatgpt",
    "NLP",
    "agentic ai"
]

# Save keywords_list
with open("data/keywords_list.pkl", "wb") as f:
    pickle.dump(keywords_list, f)
import torch.nn.functional as F
from models import BERTEmbeddings

# Input sentences
sentence1 = "BERT is an amazing model for natural language processing."
sentence2 = "a"

model = BERTEmbeddings()

# Generate embeddings
embeddings = model([sentence1, sentence2])


print(f"Embeddingfs: {embeddings.shape}")

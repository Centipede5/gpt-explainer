import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Encode text
sentence = "Bats are the only mammals capable of true sustained flight"
inputs = tokenizer(sentence, return_tensors='pt')

# Load pre-trained model
model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=True)

# Forward pass, get hidden states and attentions
outputs = model(**inputs)
attentions = outputs.attentions

# Get the attention matrix for the first layer
attention_matrix = attentions[0][0].detach().numpy()

# Print the attention matrix
print("Attention matrix for each word pair:")
print(attention_matrix)
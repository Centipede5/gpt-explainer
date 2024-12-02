from transformers import pipeline
unmasker = pipeline('fill-mask', model='distilbert-base-uncased',top_k=100)

emotions = ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'neutral']
import pickle
import torch
import pandas as pd
from main import CNN_Text, predict_single
from keras.preprocessing.text import Tokenizer

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('labelencoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

model = torch.load("./textcnn_model")
model.eval()
text = "I can't sleep at nights. I take a lot of pills but none of them helped me."
outcome = predict_single(text, model, tokenizer, le)
print(outcome)
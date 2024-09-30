from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow
from keras.models import Sequential
import keras.utils as ku
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import streamlit as st
from numpy.random import seed


import pandas as pd
import numpy as np
import string, os 

file = open("as.txt",encoding='utf-8').read()
from nltk.tokenize import sent_tokenize, word_tokenize

from functools import lru_cache


@lru_cache(maxsize=None)
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

corpus = [clean_text(x) for x in file.splitlines() if x.strip()]  # Avoid empty lines
print(corpus[:10])
tokenizer = Tokenizer()
def get_sequence_of_tokens(corpus):
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(len(token_list)-1):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
print(inp_sequences[:10])
def generate_padded_sequences(input_sequences):
    if not input_sequences:  # Check if the list is empty
        return None, None, None

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

# Tokenization
inp_sequences, total_words = get_sequence_of_tokens(corpus)


# Generate padded sequences
predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

if os.path.exists('text_generation_model.h5'):
    print("Loading pre-trained model...")
    model = load_model('text_generation_model.h5')
else:
    print("Training the model...")
    model = create_model(max_sequence_len, total_words)
    model.summary()
    model.fit(predictors, label, epochs=100, verbose=1)
    model.save('text_generation_model.h5')
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted = np.random.choice(len(predicted_probs), p=predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()

st.title("Real-time Text Generation")
seed_text = st.text_input("Start typing your sentence here:")
Number=int(st.number_input("Enter the size of the text"))
if seed_text:
    next_word = generate_text(seed_text,Number, model, max_sequence_len)
    st.write(f"Generated text: {seed_text} {next_word}")
import os
import json
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BartTokenizer, BartForConditionalGeneration

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=64, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(train_data, train_labels):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_data)
    sequences = tokenizer.texts_to_sequences(train_data)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    model = create_model()
    
    model.fit(padded_sequences, np.array(train_labels), epochs=10, validation_split=0.1)

    return model, tokenizer

# Replace with your actual sentences and labels
sentences = ["Your actual sentence 1", "Another sentence"]
labels = [1, 0]

model, tokenizer = train_model(sentences, labels)

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the model
model.save('/Users/david/Desktop/skanzzz/model')

# Load the saved model and tokenizer
model = load_model('/Users/david/Desktop/skanzzz/model')

# ...The earlier portion of your script...

# Load the saved model and tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def process_file(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        tokenized_content = tokenizer.texts_to_sequences([content])
        padded_content = pad_sequences(tokenized_content, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        prediction = model.predict(padded_content)[0][0]
        return "Category A" if prediction > 0.5 else "Category B"
    except Exception as e:
        print(f"Failed to process {filepath} due to {e}")
        return None

# Load Bart model
MODEL = 'facebook/bart-large-cnn'
tokenizer_bart = BartTokenizer.from_pretrained(MODEL)
model_bart = BartForConditionalGeneration.from_pretrained(MODEL)

def llm_generate_summary(filepath):
    try:
        with open(filepath, 'r') as file:
            text = file.read()
        inputs = tokenizer_bart([text], max_length=1024, return_tensors='pt')
        summary_ids = model_bart.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
        summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Failed to generate summary due to {e}")
        return None

def scan_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            prediction = process_file(filepath)
            summary = llm_generate_summary(filepath)
            print(f'Filename: {file}, Prediction: {prediction}, Summary: {summary}')

# Specify the directory to be scanned
scan_files('/Users/david/Desktop/skanzzz')
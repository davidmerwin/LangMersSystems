from keras.preprocessing.text import Tokenizer
# Load a saved mode
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BartTokenizer, BartForConditionalGeneration

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibilitimport os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BartTokenizer, BartForConditionalGeneration
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Constants
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

# Create LSTM model
def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=64, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train LSTM model
def train_model(train_data, train_labels):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_data)
    sequences = tokenizer.texts_to_sequences(train_data)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    model = create_model()

    try:
        model.fit(padded_sequences, np.array(train_labels), epochs=10, validation_split=0.1)
        model.save('/path/to/save/model')
        with open('/path/to/save/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Training or saving failed: {str(e)}")

    return model, tokenizer

# Placeholder data
sentences = ["This is a sentence", "Another sentence"]
labels = [1, 0]

# Train the model
model, tokenizer = train_model(sentences, labels)

def rename_file(old_name, tag, dir_path):
    base_name, ext = os.path.splitext(old_name)
    new_name = f"{base_name}_{tag}{ext}"
    os.rename(os.path.join(dir_path, old_name), os.path.join(dir_path, new_name))
# Constants
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

# Create LSTM model
def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=64, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train LSTM model
def train_model(train_data, train_labels):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_data)
    sequences = tokenizer.texts_to_sequences(train_data)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    model = create_model()
    model.fit(padded_sequences, np.array(train_labels), epochs=10, validation_split=0.1)
    return model, tokenizer

# Placeholder data
sentences = ["This is a sentence", "Another sentence"]
labels = [1, 0]
def scan_files(directory, save=False):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(f"Processing file {count + 1}/{len(files)}.")
            filepath = os.path.join(root, file)
            prediction = process_file(filepath)
            summary = llm_generate_summary(filepath)
            print(f"Summary: {summary}, Prediction: {prediction}")
            if save:
                try:
                    with open('result.json', 'a') as jsonfile:
                        result = {'filename': file, 'filepath': filepath, 'prediction': prediction, 'summary': summary}
                        json.dump(result, jsonfile)
                    rename_file(file, prediction, root)  # Rename file after processing
                except Exception as e:
                    print(f"Failed to save the result due to {str(e)}")
            count += 1
            print("-----------------")
# Train the model
model, tokenizer = train_model(sentences, labels)

# D-Wave setup
sampler = EmbeddingComposite(DWaveSampler())
x, y, z = dimod.Binary('x'), dimod.Binary('y'), dimod.Binary('z')
bqm = dimod.BQM({x: 2, y: 2, z: 2}, {(x, y): -1, (x, z): -1}, 0, 'BINARY')
sampleset = sampler.sample(bqm, num_reads=100)
print("D-Wave sampleset:", sampleset)

# Process file with NLP
def process_file_NLP(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        tokenized_content = tokenizer.texts_to_sequences([content])
        padded_content = pad_sequences(tokenized_content, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        prediction = model.predict([padded_content])[0][0]
        return prediction > 0.5
    except Exception as e:
        print(f"Failed to process {filepath} due to {e}")
        return None

# BART summarization
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
        print(f"Failed to generate summary for {filepath} due to {e}")
        return None

# Scan files and process
def scan_files(directory):
    results = {}
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            prediction = process_file_NLP(filepath)
            summary = llm_generate_summary(filepath)
            results[file] = {'prediction': prediction, 'summary': summary}
            print(f"Processed {file}. Prediction: {prediction}, Summary: {summary}")

    # Save results to JSON
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
try:
    # Your code here
except Exception as e:
    print(f"An error occurred: {str(e)}")

model.save('/path/to/save/model')
from tensorflow.keras.models import load_model
import pickle

with open('/path/to/save/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

try:
    model = load_model('/path/to/save/model')
except Exception as e:
    print(f"Couldn't load the model: {str(e)}")

# Constants
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

# Function to create an LSTM model
def create_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=64, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Replace with your actual sentences and labels
sentences = ["Your actual sentence 1", "Your actual sentence 2"]
labels = [1, 0]

# Function to train the LSTM model
def train_model(train_data, train_labels, reliable=False):
    if not reliable:
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(train_data)
        sequences = tokenizer.texts_to_sequences(train_data)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
        model = create_model()
        model.fit(padded_sequences, np.array(train_labels), epochs=10, validation_split=0.1)
    else:
        # Replace with the path to your actual model and tokenizer
        model = tf.keras.models.load_model('/path/to/your/model')
        with open('/path/to/your/tokenizer.json') as f:
            data = json.load(f)
            tokenizer = Tokenizer.from_json(data)
    return model, tokenizer

# Train the LSTM model
model, tokenizer = train_model(sentences, labels)

# Function to process each file
def process_file(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        tokenized_content = tokenizer.texts_to_sequences([content])
        padded_content = pad_sequences(tokenized_content, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        prediction = model.predict([padded_content])[0][0]
        return "Category A" if prediction > 0.5 else "Category B"
    except Exception as e:
        print(f"Failed to process file {filepath} due to {str(e)}")
        return None

# Function to generate summary using BART LLM
def llm_generate_summary(filepath):
    MODEL = 'facebook/bart-large-cnn'
    tokenizer_bart = BartTokenizer.from_pretrained(MODEL)
    model_bart = BartForConditionalGeneration.from_pretrained(MODEL)
    try:
        with open(filepath, 'r') as file:
            text = file.read()
        inputs = tokenizer_bart([text], max_length=1024, return_tensors='pt')
        summary_ids = model_bart.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
        summary = [tokenizer_bart.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        return summary[0]
    except Exception as e:
        print(f"Failed to generate summary for {filepath} due to {str(e)}")
        return None

# Function to scan files and process
def scan_files(directory, save=False):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(f"Processing file {count + 1}/{len(files)}.")
            filepath = os.path.join(root, file)
            prediction = process_file(filepath)
            summary = llm_generate_summary(filepath)
            print(f"Summary: {summary}, Prediction: {prediction}")
            if save:
                try:
                    with open('result.json', 'a') as jsonfile:
                        result = {'filename': file, 'filepath': filepath, 'prediction': prediction, 'summary': summary}
                        json.dump(result, jsonfile)
                except Exception as e:
                    print(f"Failed to save the result due to {str(e)}")
            count += 1
            print("-----------------")

# Replace with the actual path to your directory
scan_files('/Users/david/Desktop/skanzzz', save=True)
y
numpy.random.seed(7)
# load our data
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create our model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# Save our model
model.save('/Users/david/Desktop/skanzzz')

# Load a saved model and tokenizer
try:
    model = tf.keras.models.load_model('/path/to/model') # replace with the path to your actual model
    with open('/path/to/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Couldn't load the model or tokenizer: {str(e)}")
    # Handle exception...l
from keras.models import load_model
my_loaded_model = load_model('/path/to/my_model')
with open('/path/to/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)
# Save our tokenizer
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences) # Replace `sentences` with your actual data
import pickle

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL
        
                # Main function
    if __name__ == "__main__":
        scan_files('/Users/david/Desktop/skanzzz')  # Replace with your directory)
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
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

sentences = ["This is a sentence", "Another sentence"]
labels = [1, 0]

def train_model(train_data, train_labels,reliable = False):
    if not reliable: 
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<OOV>')
        tokenizer.fit_on_texts(train_data)
        sequences = tokenizer.texts_to_sequences(train_data)
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
        model = create_model()
        model.fit(padded_sequences, np.array(train_labels), epochs=10, validation_split=0.1)
    else:
        model = tf.keras.models.load_model('/path/to/model')
        with open('/path/to/tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
    return model, tokenizer

model, tokenizer = train_model(sentences, labels)

def process_file_NLP(filepath):
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        tokenized_content = tokenizer.texts_to_sequences([content])
        padded_content = pad_sequences(tokenized_content, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        prediction = model.predict([padded_content])
        return prediction.tolist()[0][0]>0.5
    except Exception as e:
        print(f"Failed to process file {filepath} due to {str(e)}")
        return None

MODEL = 'facebook/bart-large-cnn'
tokenizer_bart = BartTokenizer.from_pretrained(MODEL)
model_bart = BartForConditionalGeneration.from_pretrained(MODEL)

def llm_generate_summary(filepath):
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

def scan_files(directory,process_file,llm_generate_summary,save=False):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(f"Processing file {count}/{len(files)}.")
            filepath = os.path.join(root, file)
            prediction = process_file(filepath)
            summary = llm_generate_summary(filepath)
            print(f"Summary: {summary}, prediction: {prediction}")
            if save:
                with open('result.json', 'a') as jsonfile:
                    try:
                        result = {'filename': file,'filepath': filepath,'prediction': prediction,'summary': summary}
                        json.dump(result,jsonfile)
                    except Exception as e:
                        print(f"Failed to save the result due to {str(e)}")
            count += 1
            print("-----------------")

scan_files('/path/to/your/folder',process_file_NLP,llm_generate_summary)
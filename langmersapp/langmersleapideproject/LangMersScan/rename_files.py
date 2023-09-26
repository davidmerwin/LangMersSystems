import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Train a simple Keras model (For demonstration purposes only)
def train_model():
    file_names = ['document.txt', 'image.jpg', 'audio.mp3', 'video.mp4', 'data.json', 'picture.png', 'file.pdf', 'archive.tar', 'doc.docx', 'script.py']
    labels = list(range(10))  # One label for each file type

    tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
    tokenizer.fit_on_texts(file_names)

    sequences = tokenizer.texts_to_sequences(file_names)
    padded_sequences = pad_sequences(sequences, maxlen=5, padding='post')

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(100, 16, input_length=5),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Now predicting 10 classes
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(padded_sequences, np.array(labels), epochs=50)

    return model, tokenizer

def tag_file(model, tokenizer, filename):
    sequences = tokenizer.texts_to_sequences([filename])
    padded = pad_sequences(sequences, maxlen=5, padding='post')
    prediction = model.predict(padded)
    tag = np.argmax(prediction)
    return str(tag)
def tag_file(model, tokenizer, filename):
    sequences = tokenizer.texts_to_sequences([filename])
    padded = pad_sequences(sequences, maxlen=5, padding='post')
    prediction = model.predict(padded)
    tag = np.argmax(prediction)
    return str(tag)

    def llm_generate_comment(tag):
        tag_to_type = {
            "0": "Text File",
            "1": "Image (JPEG)",
            "2": "Audio (MP3)",
            "3": "Video (MP4)",
            "4": "JSON Data",
            "5": "Image (PNG)",
            "6": "PDF Document",
            "7": "Tar Archive",
            "8": "Word Document",
            "9": "Python Script"
        }
        return f"This is a {tag_to_type.get(tag, 'Unknown')}."

    # renamed files is not implemented too, so created function for it.
    def rename_file(file_path, tag, comment):
        base_dir, old_file_name = os.path.split(file_path)
        file_extension = old_file_name.split('.')[-1]
        new_file_name = f"{tag}_{comment[:10]}_{old_file_name}.{file_extension}"
        os.rename(file_path, os.path.join(base_dir, new_file_name))    
def llm_generate_comment(tag):
    # This is a simplified "logical language model" (LLM) for demonstration.
    # Replace with a more sophisticated model for actual use.
    tag_to_type = {
        "0": "Text File",
        "1": "Image (JPEG)",
        "2": "Audio (MP3)",
        "3": "Video (MP4)",
        "4": "JSON Data",
        "5": "Image (PNG)",
        "6": "PDF Document",
        "7": "Tar Archive",
        "8": "Word Document",
        "9": "Python Script"
    }
    return f"This is a {tag_to_type.get(tag, 'Unknown')}."

def main():
    model, tokenizer = train_model()
    comment_dict = {}

    directory = "/Users/username/your_directory"  # Replace with the path to your folder

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)

            # Skip directories
            if os.path.isdir(file_path):
                continue

            tag = tag_file(model, tokenizer, filename)
            comment = llm_generate_comment(tag)
            comment_dict[filename] = comment

            rename_file(file_path, tag)

    # Save comments to a JSON file
    with open("/Users/username/your_directory/file_comments.json", "w") as f:
        json.dump(comment_dict, f, indent=4)

if __name__ == "__main__":
    main()
    
import json
import random
import numpy as np

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, TextVectorization, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D

# Read the json file data
with open('intents - Copy.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

intents = data['intents']

words, tags = [], []

# Get patterns and tags
for intent in intents:
    for pattern in intent['patterns']:
        words.append(pattern.lower())
        tags.append(intent['tag'])

# Sort out tags and index them
uniquetags = sorted(set(tags))
tagByIndex = {index: tag for tag, index in enumerate(uniquetags)}
y = [tagByIndex[i] for i in tags]

# Create the text vectorisation
vectoriser = TextVectorization(max_tokens=1000, output_mode='int', output_sequence_length=20)
vectoriser.adapt(words)

# Build the model
tokensMax = 20000

input = tf.keras.Input(shape=(), dtype=tf.string)
x = vectoriser(input)
x = Embedding(tokensMax, 32)(x)
x = GlobalAveragePooling1D()(x) # Take average of words to create a summary vector
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(len(uniquetags), activation='softmax')(x)
model = tf.keras.Model(input, output)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Make them into arrays
words = np.array(words, dtype=np.object_)
y = np.array(y, dtype=np.int32)

# Train the model
model.fit(words, y, epochs=10, batch_size=32, validation_split=0.2, shuffle=True)

model.save('chat_intent_model.h5')

print("Model trained and saved as 'chat_intent_model.h5'")

# Predict the tag and get the score of the model
def predict_class(input_text):
    input_text = np.array([input_text], dtype=np.object_)
    predictions = model.predict(input_text)[0]
    index = int(np.argmax(predictions))
    score = np.max(predictions) * 100
    print(f"Predicted score: {score}")
    return uniquetags[index], score

# Display the tag and responses
predicted_tag, score = predict_class("Who invented computers")
for i in intents:
    if i['tag'] == predicted_tag:
        print(f"Predicted tag: {predicted_tag} -> {random.choice(i['responses'])}")
# if score <= 10 :
#     print("unknown tag")

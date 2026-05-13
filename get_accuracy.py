import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

print("Loading data...")
data = pd.read_csv("IMDB_Dataset.csv")
data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)
train_data, test_data = train_test_split(data, test_size = 0.2, random_state=42)

print("Recreating Tokenizer...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])

print("Preprocessing test data...")
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)
Y_test = test_data["sentiment"]

print("Rebuilding model architecture...")
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))
model.build(input_shape=(None, 200))

print("Loading weights...")
model.load_weights("model.h5")

print("Compiling model...")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print(f"Accuracy: {accuracy}")

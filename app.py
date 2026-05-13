import os
import sys
from tf_keras.models import load_model
import joblib
from tf_keras.preprocessing.sequence import pad_sequences
import gradio as gr

import tf_keras.src.preprocessing.text
sys.modules['keras.src.preprocessing.text'] = sys.modules['tf_keras.src.preprocessing.text']
sys.modules['keras.src.preprocessing'] = sys.modules['tf_keras.src.preprocessing']

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_dir, "model.h5")
tokenizer_path = os.path.join(script_dir, "tokenizer.pkl")

model = load_model(model_path)
tokenizer = joblib.load(tokenizer_path)

def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

title = "MOVIE SENTIMENT ANALYSIS APPLICATION"

app = gr.Interface(fn=predictive_system, inputs="textbox", outputs="textbox", title=title)

if __name__ == "__main__":
    app.launch(inbrowser=True)

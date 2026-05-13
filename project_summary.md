# Project Summary: IMDB Movie Review Sentiment Analysis

This document provides a summary of the work done in the IMDB Movie Review Sentiment Analysis project, including descriptions of the files and functions implemented.

## 1. File Structure and Purpose

*   **`IMDB_Movie_Review_Sentiment_Analysis.ipynb` / `IMDB_Movie_Review_Sentiment_Analysis.py`**:
    *   **Purpose**: These files contain the core machine learning pipeline for data preprocessing, model building, training, and evaluation.
    *   **Work Done**:
        *   Loads the IMDB dataset (`IMDB_Dataset.csv`).
        *   Preprocesses text data by replacing string labels ("positive", "negative") with numerical values (1, 0).
        *   Splits data into training and testing sets.
        *   Tokenizes the text data using TensorFlow Keras `Tokenizer` (limiting to the top 5000 words).
        *   Pads the sequences to a maximum length of 200 to ensure uniform input size.
        *   Builds an LSTM (Long Short-Term Memory) neural network model using `Sequential`, `Embedding`, `LSTM`, and `Dense` layers.
        *   Compiles and trains the model for 5 epochs.
        *   Saves the trained model as `model.h5`.
        *   Saves the tokenizer object as `tokenizer.pkl` using `joblib`.
        *   Evaluates the model accuracy and loss on the test set.

*   **`app.py`**:
    *   **Purpose**: This script serves as the backend and frontend application using Gradio. It allows users to interact with the trained sentiment analysis model through a web interface.
    *   **Work Done**:
        *   Loads the saved model (`model.h5`) and tokenizer (`tokenizer.pkl`).
        *   Defines the prediction logic.
        *   Sets up a Gradio web interface (`gr.Interface`) with a text box input and a text box output.
        *   Launches the web application.

*   **`Model_Testing_&_Web_Application.ipynb`**:
    *   **Purpose**: A Jupyter notebook dedicated to loading the saved model and tokenizer, testing it with custom inputs, and prototyping the web application interface before moving it to the final `app.py` script.

*   **`model.h5`**:
    *   **Purpose**: The saved weights and architecture of the trained LSTM neural network model.

*   **`tokenizer.pkl`**:
    *   **Purpose**: The saved Tokenizer object, necessary to convert new, unseen text into the sequence of integers that the model expects, using the exact same vocabulary index it learned during training.

## 2. Key Functions Implemented

### `predictive_system(review)`
*   **Location**: Found in both `app.py` and `IMDB_Movie_Review_Sentiment_Analysis.py` (and corresponding notebooks).
*   **Parameters**:
    *   `review` (str): A string containing the text of a movie review.
*   **Process**:
    1.  Uses the loaded `tokenizer` to convert the input `review` text into a sequence of integers (`texts_to_sequences`).
    2.  Pads the sequence using `pad_sequences` to ensure its length is exactly 200, matching the input shape the model was trained on.
    3.  Passes the padded sequence to `model.predict()` to obtain a prediction score between 0 and 1.
    4.  Applies a threshold: if the score is > 0.5, it classifies it as "positive"; otherwise, "negative".
*   **Returns**: A string indicating the sentiment, either `"positive"` or `"negative"`.

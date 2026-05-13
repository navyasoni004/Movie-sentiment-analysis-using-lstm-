# IMDB Movie Review Sentiment Analysis using LSTM

This project is a Deep Learning application that classifies movie reviews as either **Positive** or **Negative** using a Long Short-Term Memory (LSTM) network. It includes a web interface built with Gradio for real-time sentiment prediction.

## 🚀 Features
- **LSTM Neural Network**: Built with TensorFlow/Keras for sequence modeling.
- **Natural Language Processing**: Custom tokenization and padding for text preprocessing.
- **Interactive Web App**: Uses Gradio to provide a simple UI for testing reviews.
- **Evaluation Metrics**: Includes scripts to calculate Accuracy and F1-score.

## 🛠️ Tech Stack
- **Language**: Python
- **Libraries**: TensorFlow, Keras, Pandas, Scikit-learn, Gradio
- **Model**: LSTM (Long Short-Term Memory)

## 📁 Project Structure
- `app.py`: The main entry point for the Gradio web application.
- `model.h5`: The trained LSTM model weights.
- `tokenizer.pkl`: Pickled tokenizer used for text preprocessing.
- `get_accuracy.py`: Script to evaluate the model's accuracy on the dataset.
- `get_f1.py`: Script to calculate the F1-score of the model.
- `IMDB_Movie_Review_Sentiment_Analysis.ipynb`: Notebook containing the training process.

## ⚙️ How to Run

1. **Install Dependencies**:
   ```bash
   pip install tensorflow pandas scikit-learn gradio
   ```

2. **Run the Web App**:
   ```bash
   python app.py
   ```

3. **Evaluate the Model**:
   ```bash
   python get_accuracy.py
   ```

## 📊 Dataset
The model was trained on the [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). (Note: The dataset file is ignored in this repository due to its size).

---
*Created by [Navya Soni](https://github.com/navyasoni004)*
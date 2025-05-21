import pickle
import numpy as np
from manualSVM import clean_text, preprocess, vectorize_tfidf, LinearSVM

# Load the model
with open("svm_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    print(model_data.keys())

model = model_data['model']
word2idx = model_data['word2idx']
idf = model_data['idf']
label2idx = model_data['label2idx']
idx2label = model_data['idx2label']
ID_STOPWORDS = model_data['stopwords']

def predict_emotion(text):
    cleaned = clean_text(text)
    tokens = preprocess(cleaned)
    vector = vectorize_tfidf(tokens, word2idx, idf)
    # prediction = model.predict(np.array([vector]))[0]
    prediction = model.predict(np.array([vector]))[0]
    return idx2label[prediction]

import tkinter as tk
from tkinter import messagebox

def on_predict():
    input_text = text_input.get("1.0", "end-1c")
    if not input_text.strip():
        messagebox.showwarning("Warning", "Please enter some text.")
        return
    result = predict_emotion(input_text)
    result_label.config(text=f"Predicted Emotion: {result}")

# Create GUI
root = tk.Tk()
root.title("Emotion Predictor (SVM)")
root.geometry("400x300")

tk.Label(root, text="Enter Text:", font=("Arial", 12)).pack(pady=5)
text_input = tk.Text(root, height=5, width=40)
text_input.pack(pady=5)

predict_btn = tk.Button(root, text="Predict Emotion", command=on_predict)
predict_btn.pack(pady=10)

result_label = tk.Label(root, text="Predicted Emotion: -", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

root.mainloop()

import numpy as np
import pickle
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Redefine the LinearSVM and OneVsRestSVM classes ---

class LinearSVM:
    def __init__(self, num_features, lr=0.1, C=1, epochs=10):
        self.w = np.zeros(num_features)
        self.b = 0
        self.lr = lr
        self.C = C
        self.epochs = epochs

    def fit(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                xi, yi = X[i], y[i]
                condition = yi * (np.dot(self.w, xi) + self.b) >= 1
                if condition:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.C * yi * xi)
                    self.b += self.lr * self.C * yi

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

class OneVsRestSVM:
    def __init__(self, num_classes, num_features, lr=0.1, C=1, epochs=10):
        self.models = [LinearSVM(num_features, lr, C, epochs) for _ in range(num_classes)]

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            binary_y = np.where(y == i, 1, -1)
            model.fit(X, binary_y)

    def predict(self, X):
        scores = [np.dot(X, model.w) + model.b for model in self.models]
        return np.argmax(scores, axis=0)

# --- Load trained model ---
with open('svm_model1_tuned.pkl', 'rb') as f:
    model_data = pickle.load(f)

models = model_data['models']  # already trained LinearSVMs
vocab = model_data['vocab']
word2idx = model_data['word2idx']
emotion2idx = model_data['emotion2idx']
idx2emotion = {idx: emotion for emotion, idx in emotion2idx.items()}

# --- Preprocessing (must match training preprocessing) ---
factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def vectorize(tokens):
    vec = np.zeros(len(vocab))
    for word in tokens:
        if word in word2idx:
            vec[word2idx[word]] += 1
    return vec

# --- Predict ---
def predict(text):
    tokens = preprocess(text)
    vec = vectorize(tokens)
    scores = [np.dot(vec, model.w) + model.b for model in models]
    predicted_idx = np.argmax(scores)
    return idx2emotion[predicted_idx]

# --- Example ---
text_input = "Aku sangat senang bertemu kamu hari ini!"
predicted_emotion = predict(text_input)
print(f"Predicted Emotion: {predicted_emotion}")

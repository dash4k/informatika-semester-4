import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm import tqdm
from itertools import product
import pickle

# --- Load & Label ---
def load_data():
    emotions = ['Anger', 'Fear', 'Joy', 'Love', 'Neutral', 'Sad']
    all_data = []

    for emotion in emotions:
        df = pd.read_csv(f"dataset/{emotion}Data.csv", sep='\t', on_bad_lines='skip')
        df['Label'] = emotion
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

data = load_data()

factory = StopWordRemoverFactory()
stop_words = set(factory.get_stop_words())

# --- Preprocessing ---
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

data['tokens'] = data['Tweet'].apply(preprocess)

# --- Vocabulary ---
vocab = sorted(set(word for tokens in data['tokens'] for word in tokens))
word2idx = {word: idx for idx, word in enumerate(vocab)}


def vectorize(tokens):
    vec = np.zeros(len(vocab))
    for word in tokens:
        if word in word2idx:
            vec[word2idx[word]] += 1
    return vec

data['vector'] = data['tokens'].apply(vectorize)

# --- Labels ---
emotions = sorted(data['Label'].unique())
emotion2idx = {emotion: idx for idx, emotion in enumerate(emotions)}

# Final label mapping
data['y'] = data['Label'].map(emotion2idx)
X = np.stack(data['vector'].values)
y = data['y'].values

# Split into train/test (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- SVM Classes ---
class LinearSVM:
    def __init__(self, num_features, lr=0.1, C=1, epochs=10):
        self.w = np.zeros(num_features)
        self.b = 0
        self.lr = lr
        self.C = C
        self.epochs = epochs

    def fit(self, X, y):
        for _ in tqdm(range(self.epochs), desc="Training SVM", leave=False):
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
        for i, model in enumerate(tqdm(self.models, desc="Training One-vs-Rest SVM")):
            binary_y = np.where(y == i, 1, -1)
            model.fit(X, binary_y)

    def predict(self, X):
        scores = [np.dot(X, model.w) + model.b for model in self.models]
        return np.argmax(scores, axis=0)

# # --- Cross Validation ---
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# accuracies = []
# all_preds, all_true = [], []

# for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_trainval), total=kf.get_n_splits(), desc="Cross-validation Folds")):
#     X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
#     y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

#     model = OneVsRestSVM(num_classes=len(emotions), num_features=X.shape[1], lr=0.01, C=0.1, epochs=10)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_val)

#     acc = np.mean(preds == y_val)
#     accuracies.append(acc)
#     all_preds.extend(preds)
#     all_true.extend(y_val)

# print(f"\nAverage CV Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

# # --- Final Evaluation on Unseen Test Set ---
# final_model = OneVsRestSVM(num_classes=len(emotions), num_features=X.shape[1], lr=0.001, C=0.01, epochs=10)
# final_model.fit(X_trainval, y_trainval)
# test_preds = final_model.predict(X_test)
# test_acc = np.mean(test_preds == y_test)
# print(f"\nTest Accuracy (20% holdout): {test_acc:.4f}")

# # --- Confusion Matrix ---
# cm = confusion_matrix(y_test, test_preds)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix (Test Set)")
# plt.show()

# # Save model
# with open('svm_model1.pkl', 'wb') as f:
#     pickle.dump({
#         'models': final_model.models,
#         'vocab': vocab,
#         'word2idx': word2idx,
#         'emotion2idx': emotion2idx,
#     }, f)
    
# # --- Binary Labels for Emotion Detection ---
# label = 'Neutral'

# data['y_binary'] = data['Label'].apply(lambda x: 1 if x == label else -1)

# X = np.stack(data['vector'].values)
# y = data['y_binary'].values

# # --- Cross Validation for Anger Only ---
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# accuracies = []
# all_preds, all_true = [], []

# for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=kf.get_n_splits(), desc="Cross-validation Folds")):
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     model = LinearSVM(num_features=X.shape[1], lr=0.01, C=0.1, epochs=10)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_val)

#     acc = np.mean(preds == y_val)
#     accuracies.append(acc)
#     all_preds.extend(preds)
#     all_true.extend(y_val)

# print(f"\n\nAverage Accuracy (Anger vs Others): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

# # --- Confusion Matrix ---
# cm = confusion_matrix(all_true, all_preds, labels=[1, -1])
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[label, 'Not ' + label], yticklabels=[label, 'Not ' + label])
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title(f"Confusion Matrix ({label} Detection)")
# plt.show()

# # --- Save Model ---
# with open('svm_anger.pkl', 'wb') as f:
#     pickle.dump({
#         'model': model,
#         'vocab': vocab,
#         'word2idx': word2idx,
#     }, f)
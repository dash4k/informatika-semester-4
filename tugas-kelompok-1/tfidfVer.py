# import numpy as np
# import pandas as pd
# import re
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# from tqdm import tqdm
# from itertools import product
# import pickle

# # --- Load & Label ---
# def load_data():
#     emotions = ['Anger', 'Fear', 'Joy', 'Love', 'Neutral', 'Sad']
#     all_data = []

#     for emotion in emotions:
#         df = pd.read_csv(f"dataset/{emotion}Data.csv", sep='\t', on_bad_lines='skip')
#         df['Label'] = emotion
#         all_data.append(df)

#     return pd.concat(all_data, ignore_index=True)

# data = load_data()

# factory = StopWordRemoverFactory()
# stop_words = set(factory.get_stop_words())

# # --- Preprocessing ---
# def preprocess(text):
#     text = str(text).lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     tokens = text.split()
#     tokens = [word for word in tokens if word not in stop_words]
#     return tokens

# data['tokens'] = data['Tweet'].apply(preprocess)

# # --- Vocabulary ---
# vocab = sorted(set(word for tokens in data['tokens'] for word in tokens))
# word2idx = {word: idx for idx, word in enumerate(vocab)}


# def vectorize(tokens):
#     vec = np.zeros(len(vocab))
#     for word in tokens:
#         if word in word2idx:
#             vec[word2idx[word]] += 1
#     return vec

# data['vector'] = data['tokens'].apply(vectorize)

# # --- Labels ---
# emotions = sorted(data['Label'].unique())
# emotion2idx = {emotion: idx for idx, emotion in enumerate(emotions)}

# # Final label mapping
# data['y'] = data['Label'].map(emotion2idx)
# X = np.stack(data['vector'].values)
# y = data['y'].values

# # Split into train/test (80/20)
# X_trainval, X_test, y_trainval, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # --- SVM Classes ---
# class LinearSVM:
#     def __init__(self, num_features, lr=0.1, C=1, epochs=10):
#         self.w = np.zeros(num_features)
#         self.b = 0
#         self.lr = lr
#         self.C = C
#         self.epochs = epochs

#     def fit(self, X, y):
#         for _ in tqdm(range(self.epochs), desc="Training SVM", leave=False):
#             for i in range(len(X)):
#                 xi, yi = X[i], y[i]
#                 condition = yi * (np.dot(self.w, xi) + self.b) >= 1
#                 if condition:
#                     self.w -= self.lr * self.w
#                 else:
#                     self.w -= self.lr * (self.w - self.C * yi * xi)
#                     self.b += self.lr * self.C * yi

#     def predict(self, X):
#         return np.sign(np.dot(X, self.w) + self.b)

# class OneVsRestSVM:
#     def __init__(self, num_classes, num_features, lr=0.1, C=1, epochs=10):
#         self.models = [LinearSVM(num_features, lr, C, epochs) for _ in range(num_classes)]

#     def fit(self, X, y):
#         for i, model in enumerate(tqdm(self.models, desc="Training One-vs-Rest SVM")):
#             binary_y = np.where(y == i, 1, -1)
#             model.fit(X, binary_y)

#     def predict(self, X):
#         scores = [np.dot(X, model.w) + model.b for model in self.models]
#         return np.argmax(scores, axis=0)
    
# # --- Define Hyperparameter Grid ---
# param_grid = {
#     'lr': [0.01, 0.001],
#     'C': [0.1, 1],
#     'epochs': [5, 10]
# }

# param_combinations = list(product(param_grid['lr'], param_grid['C'], param_grid['epochs']))

# best_score = 0
# best_params = None

# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# print("Starting hyperparameter tuning...")

# # --- Hyperparameter Search ---
# for lr, C, epochs in tqdm(param_combinations, desc="Hyperparameter combinations"):
#     fold_accuracies = []

#     for train_idx, val_idx in kf.split(X_trainval):
#         X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
#         y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

#         model = OneVsRestSVM(num_classes=len(emotions), num_features=X.shape[1], lr=lr, C=C, epochs=epochs)
#         model.fit(X_train, y_train)
#         preds = model.predict(X_val)

#         acc = np.mean(preds == y_val)
#         fold_accuracies.append(acc)

#     avg_acc = np.mean(fold_accuracies)

#     print(f"Params: lr={lr}, C={C}, epochs={epochs} -> Avg CV Acc = {avg_acc:.4f}")

#     if avg_acc > best_score:
#         best_score = avg_acc
#         best_params = (lr, C, epochs)

# print(f"\nBest Hyperparameters: lr={best_params[0]}, C={best_params[1]}, epochs={best_params[2]} with Avg CV Acc = {best_score:.4f}")

# # --- Final Training with Best Parameters ---
# final_model = OneVsRestSVM(
#     num_classes=len(emotions),
#     num_features=X.shape[1],
#     lr=best_params[0],
#     C=best_params[1],
#     epochs=best_params[2]
# )

# final_model.fit(X_trainval, y_trainval)

# test_preds = final_model.predict(X_test)
# test_acc = np.mean(test_preds == y_test)
# print(f"\nTest Accuracy (20% holdout) with tuned parameters: {test_acc:.4f}")

# # --- Confusion Matrix ---
# cm = confusion_matrix(y_test, test_preds)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix (Test Set)")
# plt.show()

# # Save model
# with open('svm_model2_tuned.pkl', 'wb') as f:
#     pickle.dump({
#         'models': final_model.models,
#         'vocab': vocab,
#         'word2idx': word2idx,
#         'emotion2idx': emotion2idx,
#     }, f)


import numpy as np
import pandas as pd
import re
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
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

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

i = 0

# --- Preprocessing ---
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    global i
    i += 1
    print(i)
    return tokens

print('stemming start')
data['tokens'] = data['Tweet'].apply(preprocess)
print('stemming done')

# --- Vocabulary and TF-IDF ---
vocab = sorted(set(word for tokens in data['tokens'] for word in tokens))
word2idx = {word: idx for idx, word in enumerate(vocab)}

# Compute IDF
N = len(data)
df_counts = np.zeros(len(vocab))

for tokens in data['tokens']:
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in word2idx:
            df_counts[word2idx[token]] += 1

idf = np.log((N + 1) / (df_counts + 1)) + 1  # Smoothing

def vectorize_tfidf(tokens):
    vec = np.zeros(len(vocab))
    token_counts = {}
    for word in tokens:
        if word in word2idx:
            token_counts[word] = token_counts.get(word, 0) + 1

    for word, count in token_counts.items():
        idx = word2idx[word]
        tf = count / len(tokens)  # term frequency
        vec[idx] = tf * idf[idx]  # tf-idf
    return vec

data['vector'] = data['tokens'].apply(vectorize_tfidf)

# --- Labels ---
emotions = sorted(data['Label'].unique())
emotion2idx = {emotion: idx for idx, emotion in enumerate(emotions)}

data['y'] = data['Label'].map(emotion2idx)
X = np.stack(data['vector'].values)
y = data['y'].values

# --- Train/test split ---
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

# --- Hyperparameter Grid ---
param_grid = {
    'lr': [0.01, 0.001],
    'C': [0.1, 1],
    'epochs': [5, 10]
}
param_combinations = list(product(param_grid['lr'], param_grid['C'], param_grid['epochs']))

best_score = 0
best_params = None

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Starting hyperparameter tuning...")

# --- Hyperparameter Search ---
for lr, C, epochs in tqdm(param_combinations, desc="Hyperparameter combinations"):
    fold_accuracies = []

    for train_idx, val_idx in kf.split(X_trainval):
        X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        model = OneVsRestSVM(num_classes=len(emotions), num_features=X.shape[1], lr=lr, C=C, epochs=epochs)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = np.mean(preds == y_val)
        fold_accuracies.append(acc)

    avg_acc = np.mean(fold_accuracies)

    print(f"Params: lr={lr}, C={C}, epochs={epochs} -> Avg CV Acc = {avg_acc:.4f}")

    if avg_acc > best_score:
        best_score = avg_acc
        best_params = (lr, C, epochs)

print(f"\nBest Hyperparameters: lr={best_params[0]}, C={best_params[1]}, epochs={best_params[2]} with Avg CV Acc = {best_score:.4f}")

# --- Final Model Training ---
final_model = OneVsRestSVM(
    num_classes=len(emotions),
    num_features=X.shape[1],
    lr=best_params[0],
    C=best_params[1],
    epochs=best_params[2]
)
final_model.fit(X_trainval, y_trainval)

test_preds = final_model.predict(X_test)
test_acc = np.mean(test_preds == y_test)
print(f"\nTest Accuracy (20% holdout) with tuned parameters: {test_acc:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()

# Save model
with open('svm_model2_tfidf_tuned.pkl', 'wb') as f:
    pickle.dump({
        'models': final_model.models,
        'vocab': vocab,
        'word2idx': word2idx,
        'idf': idf.tolist(),  # saving idf values
        'emotion2idx': emotion2idx,
    }, f)
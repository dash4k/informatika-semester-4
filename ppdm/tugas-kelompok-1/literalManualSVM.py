import numpy as np
import csv
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --- Hardcoded Indonesian Stopwords (Minimal Set) ---
ID_STOPWORDS = {
    "yang", "untuk", "dengan", "pada", "tidak", "dari", "ini", "di", "ke", "dan",
    "adalah", "itu", "saya", "kamu", "dia", "kita", "mereka", "akan", "apa", "bisa",
    "karena", "jadi", "jika", "agar", "dalam", "ada", "sudah", "belum", "lagi", "harus",
    "sangat", "banyak", "semua", "hanya", "saja", "mau", "boleh", "begitu", "lebih",
    "kurang", "seperti", "masih", "namun", "tetapi", "bukan", "bila", "oleh", "setelah",
    "sebelum", "kami", "aku", "engkau", "dirinya", "sendiri", "antar", "antara",
    "sehingga", "berupa", "terhadap", "pula", "tetap", "baik", "sambil", "tersebut",
    "selama", "seluruh", "bagai", "sekali", "supaya", "dapat", "bahwa", "kapan", "sebab",
    "sedang", "terjadi", "mungkin", "saat", "menjadi", "apakah", "dimana", "kemana"
}

# --- Load Dataset ---
def load_data():
    emotions = ['Anger', 'Fear', 'Joy', 'Love', 'Neutral', 'Sad']
    texts, labels = [], []

    for emotion in emotions:
        with open(f"dataset/{emotion}Data.csv", encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)
            for row in reader:
                if row:
                    texts.append(row[0])
                    labels.append(emotion)
    return texts, labels

# --- Text Preprocessing ---
def clean_text(text):
    text = text.lower()
    return ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)

def preprocess(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ID_STOPWORDS]
    return tokens

# --- TF-IDF Vectorization ---
def build_vocab(token_lists):
    vocab = sorted(set(word for tokens in token_lists for word in tokens))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx

def compute_idf(token_lists, word2idx):
    df_counts = np.zeros(len(word2idx))
    N = len(token_lists)
    for tokens in token_lists:
        for word in set(tokens):
            if word in word2idx:
                df_counts[word2idx[word]] += 1
    idf = np.log((N + 1) / (df_counts + 1)) + 1
    return idf

def vectorize_tfidf(tokens, word2idx, idf):
    vec = np.zeros(len(word2idx))
    token_counts = {}
    for word in tokens:
        if word in word2idx:
            token_counts[word] = token_counts.get(word, 0) + 1
    for word, count in token_counts.items():
        idx = word2idx[word]
        tf = count / len(tokens)
        vec[idx] = tf * idf[idx]
    return vec

# --- SVM Models ---
class LinearSVM:
    def __init__(self, num_features, lr=0.1, C=1, epochs=10):
        self.w = np.zeros(num_features)
        self.b = 0
        self.lr = lr
        self.C = C
        self.epochs = epochs

    def fit(self, X, y):
        for _ in tqdm(range(self.epochs), desc="Training Linear SVM", leave=False):
            for i in range(len(X)):
                xi, yi = X[i], y[i]
                if yi * (np.dot(self.w, xi) + self.b) >= 1:
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
        scores = np.array([np.dot(X, model.w) + model.b for model in self.models])
        return np.argmax(scores, axis=0)

# --- Stratified Train/Test Split ---
def stratified_split(X, y, test_size=0.2):
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0].tolist()
        random.shuffle(idx)
        split = int(len(idx) * (1 - test_size))
        train_idx += idx[:split]
        test_idx += idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# --- Manual K-Fold ---
def manual_k_fold_split(X, y, k=5):
    indices = list(range(len(X)))
    random.shuffle(indices)
    fold_size = len(X) // k
    folds = []
    for i in range(k):
        val_idx = indices[i*fold_size:(i+1)*fold_size]
        train_idx = indices[:i*fold_size] + indices[(i+1)*fold_size:]
        folds.append((train_idx, val_idx))
    return folds

# --- Confusion Matrix ---
def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

# --- MAIN PIPELINE ---
print("Loading and preprocessing data...")
texts, labels = load_data()
tokenized = [preprocess(t) for t in tqdm(texts, desc="Preprocessing")]

vocab, word2idx = build_vocab(tokenized)
idf = compute_idf(tokenized, word2idx)
X = np.array([vectorize_tfidf(t, word2idx, idf) for t in tqdm(tokenized, desc="Vectorizing")])

emotions = sorted(set(labels))
emotion2idx = {e: i for i, e in enumerate(emotions)}
y = np.array([emotion2idx[lbl] for lbl in labels])

X_trainval, X_test, y_trainval, y_test = stratified_split(X, y)

print(tokenized)
print(vocab)

# --- Hyperparameter Tuning ---
param_grid = [(lr, C, ep) for lr in [0.1, 0.01] for C in [1, 0.1] for ep in [5, 100]]
best_acc = 0
best_params = None
folds = manual_k_fold_split(X_trainval, y_trainval, k=5)

print("\nStarting hyperparameter tuning...")
for lr, C, epochs in tqdm(param_grid, desc="Param Combos"):
    fold_accs = []
    for train_idx, val_idx in folds:
        model = OneVsRestSVM(len(emotions), X.shape[1], lr, C, epochs)
        model.fit(X_trainval[train_idx], y_trainval[train_idx])
        preds = model.predict(X_trainval[val_idx])
        acc = np.mean(preds == y_trainval[val_idx])
        fold_accs.append(acc)
    avg_acc = np.mean(fold_accs)
    print(f"lr={lr}, C={C}, epochs={epochs} → CV Acc = {avg_acc:.4f}")
    if avg_acc > best_acc:
        best_acc = avg_acc
        best_params = (lr, C, epochs)

# --- Final Training ---
print(f"\nBest Params: lr={best_params[0]}, C={best_params[1]}, epochs={best_params[2]}")
final_model = OneVsRestSVM(len(emotions), X.shape[1], *best_params)
final_model.fit(X_trainval, y_trainval)

test_preds = final_model.predict(X_test)
test_acc = np.mean(test_preds == y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# --- Confusion Matrix ---
cm = confusion_matrix_np(y_test, test_preds, len(emotions))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()

# --- Save Model ---
with open('svm_model_numpy_visual.pkl', 'wb') as f:
    pickle.dump({
        'models': final_model.models,
        'vocab': vocab,
        'word2idx': word2idx,
        'idf': idf.tolist(),
        'emotion2idx': emotion2idx,
    }, f)

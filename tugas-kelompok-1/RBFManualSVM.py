import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def load_data():
    emotions = [ 'Anger', 'Fear', 'Joy', 'Love', 'Neutral', 'Sad']
    texts, labels = [], []

    for emotion in emotions:
        with open(f'dataset/{emotion}Data.csv', encoding='utf-8') as file:
            lines = file.readlines()[1:]
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if parts:
                        texts.append(parts[0])
                        labels.append(emotion)
    return texts, labels


def clean_text(text: str):
    text = text.lower()
    text = ''.join([word if word.isalnum() or word.isspace() else '' for word in text])
    text = text.strip()
    return text


def preprocess(text: str):
    tokens = text.split()
    tokens = [word for word in tokens if word not in ID_STOPWORDS]
    return tokens


def build_vocab(token_lists: list):
    vocab = sorted(set(word for tokens in token_lists for word in tokens))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx


def compute_idf(token_lists, word2idx):
    df_counts = np.zeros(len(word2idx))
    N = len(token_lists)
    for token in token_lists:
        for word in set(token):
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

    total_terms = sum(token_counts.values())

    for word, count in token_counts.items():
        idx = word2idx[word]
        tf = count / total_terms
        vec[idx] = tf * idf[idx]

    return vec

def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


class RBFKernelSVM:
    def __init__(self, gamma=0.05, C=1.0, epochs=5):
        self.gamma = gamma
        self.C = C
        self.epochs = epochs
        self.alpha = None
        self.X_train = None
        self.y_train = None

    def rbf_kernel(self, X1, X2):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            print(f"Iter: {i} from {X1.shape[0]}")
            diff = X2 - X1[i]
            K[i] = np.exp(-self.gamma * np.sum(diff ** 2, axis=1))
        return K

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.X_train = X
        self.y_train = y

        K = self.rbf_kernel(X, X)

        for epoch in range(self.epochs):
            for i in range(n_samples):
                margin = np.sum(self.alpha * y * K[:, i])
                if y[i] * margin < 1:
                    self.alpha[i] += self.C

    def predict(self, X):
        K = self.rbf_kernel(X, self.X_train)
        decision = np.dot(K, self.alpha * self.y_train)
        return np.sign(decision)


class OneVsRestKernelSVM:
    def __init__(self, num_classes, gamma=0.05, C=1.0, epochs=5):
        self.num_classes = num_classes
        self.models = [RBFKernelSVM(gamma, C, epochs) for _ in range(num_classes)]

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            binary_y = np.where(y == i, 1, -1)
            model.fit(X, binary_y)

    def predict(self, X):
        scores = []
        for model in self.models:
            decision = model.predict(X)
            scores.append(decision)
        return np.argmax(np.stack(scores, axis=1), axis=1)


import random

# Load and preprocess the data
texts, labels = load_data()
texts = [clean_text(t) for t in texts]
tokens_list = [preprocess(t) for t in texts]

# Encode labels
unique_labels = sorted(set(labels))
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label2idx[label] for label in labels])

# Reduce dataset size before splitting
portion = 0.5  # or 0.5 for 50%
reduced_size = int(len(texts) * portion)

texts = texts[:reduced_size]
tokens_list = tokens_list[:reduced_size]
y_encoded = y_encoded[:reduced_size]

# Build vocab and compute IDF
vocab, word2idx = build_vocab(tokens_list)
idf = compute_idf(tokens_list, word2idx)

# Convert texts to TF-IDF vectors
X = np.array([vectorize_tfidf(tokens, word2idx, idf) for tokens in tokens_list])
y = y_encoded

# Shuffle and split (80/20)
data = list(zip(X, y))
random.seed(42)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]

X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Train the model
num_classes = len(unique_labels)
num_features = X.shape[1]
model = OneVsRestKernelSVM(len(unique_labels), gamma=0.05, C=0.1, epochs=100)
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Compute accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix_np(y_test, predictions, len(unique_labels))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()
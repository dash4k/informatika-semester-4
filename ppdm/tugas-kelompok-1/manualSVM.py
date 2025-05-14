import numpy as np


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


class LinearSVM:
    def __init__(self, num_features, lr=0.1, C=1, epochs=10):
        self.w = np.zeros(num_features)
        self.b = 0
        self.lr = lr
        self.C = C
        self.epochs =  epochs

    def fit(self, X, y):
        for a in range(self.epochs):
            print(f"Iter: {a} from {self.epochs}")
            for i in range(len(X)):
                xi, yi = X[i], y[i]
                if yi * (np.dot(self.w, xi) + self.b) >= 1:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.C * yi * xi)
                    self.b += self.lr * self.C * yi

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    

class OneVSRestSVM:
    def __init__(self, num_classes, num_features, lr=0.1, C=1, epochs=10):
        self.models = [LinearSVM(num_features, lr, C, epochs) for _ in range(num_classes)]

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            binary_y = np.where(y == i, 1, -1)
            model.fit(X, binary_y)

    def predict(self, X):
        scores = np.array([np.dot(X, model.w) + model.b for model in self.models])
        return np.argmax(scores, axis=0)
    
import random

# Load and preprocess the data
texts, labels = load_data()
texts = [clean_text(t) for t in texts]
tokens_list = [preprocess(t) for t in texts]

# Encode labels
unique_labels = sorted(set(labels))
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label2idx[label] for label in labels])

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
model = OneVSRestSVM(num_classes, num_features, lr=0.01, C=0.5, epochs=100)
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Compute accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
from sklearn.cluster import KMeans

import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from underthesea import word_tokenize
from sklearn.decomposition import PCA
from transformers import (
    AutoTokenizer,
    AutoModel
)
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from itertools import permutations

# Đọc dữ liệu
df = pd.read_csv('synthetic_train.csv')

# df = pd.read_csv('synthetic_val.csv')

# 1. Tiền xử lý
negations = {
    "không",
    "chẳng",
    "chưa",
    "đừng"
}

def remove_english_noise(text):
    text = re.sub(r"\bthe\b", " ", text)
    text = re.sub(r"\band\b", " ", text)
    text = re.sub(r"\ba\b", " ", text)
    text = re.sub(r"\ban\b", " ", text)
    return text

def handle_negation(tokens):
    result = []
    i = 0
    while i < len(tokens):
        if (
            tokens[i] in negations
            and i + 1 < len(tokens)
        ):
            result.append(
                "NOT_" + tokens[i + 1]
            )
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

def preprocess_text(text):
    # chuẩn hóa chữ thường
    text = str(text).lower().strip()
    # loại ký tự đặc biệt
    text = remove_english_noise(text)
    text = re.sub(
        r"[^\w\sÀ-ỹ]",
        " ",
        text
    )
    # chuẩn hóa khoảng trắng
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()
    # tokenize
    text = word_tokenize(
        text,
        format="text"
    )
    tokens = text.split()
    # xử lý phủ định
    tokens = handle_negation(tokens)
    return " ".join(tokens)

print("[1/5] Đang tiền xử lý dữ liệu...")

df["clean_text"] = (
    df["sentence"]
    .astype(str)
    .apply(preprocess_text)
)

# 2. PhoBERT Embedding
# chuyển câu -> vector
print("[2/5] Đang tải PhoBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    "vinai/phobert-base"
)

phobert = AutoModel.from_pretrained(
    "vinai/phobert-base"
)

phobert.eval()

def get_embeddings_batch(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = phobert(**encoded)
        
        last_hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        embeddings = (last_hidden * mask).sum(1) / mask.sum(1)
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings)

print("[3/5] Đang tạo embedding...")

X = get_embeddings_batch(
    df["clean_text"].tolist(),
    batch_size=32
)
print(X.shape)

# 3. Phân cụm K-Means
print("[4/5] Đang phân cụm KMeans...")

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

df["Cluster"] = model.fit_predict(X)

# 4. Đánh giá
true_labels = df["sentiment"]

sentiments = [
    "negative",
    "neutral",
    "positive"
]

best_acc = 0
best_mapping = None

for perm in permutations(sentiments):
    mapping = {
        0: perm[0],
        1: perm[1],
        2: perm[2]
    }
    predicted = (
        df["Cluster"]
        .map(mapping)
    )
    acc = accuracy_score(
        true_labels,
        predicted
    )
    if acc > best_acc:
        best_acc = acc
        best_mapping = mapping

# Accuracy
print(best_mapping)

# ARI
ari = adjusted_rand_score(
    df["sentiment"],
    df["Cluster"]
)

# NMI
nmi = normalized_mutual_info_score(
    df["sentiment"],
    df["Cluster"]
)

# Silhouette
sil = silhouette_score(
    X,
    df["Cluster"]
)

print(f"Accuracy: {best_acc*100:.2f}%")
print(f"ARI: {ari*100:.2f}%")
print(f"NMI: {nmi*100:.2f}%")
print(f"Silhouette: {sil*100:.2f}%")

print(
    pd.crosstab(
        df["Cluster"],
        df["sentiment"]
    )
)

print(
    pd.crosstab(
        df["Cluster"],
        df["topic"]
    )
)

# 5. Trực quan hóa
# PCA 2D
print("[5/5] Đang vẽ biểu đồ PCA 2D...")

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(10,7))

sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=df["Cluster"].astype(str),
    alpha=0.7
)

plt.title("KMeans + PhoBERT")
plt.show()

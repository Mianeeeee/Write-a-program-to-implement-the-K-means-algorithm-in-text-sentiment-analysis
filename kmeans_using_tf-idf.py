from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from underthesea import word_tokenize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from itertools import permutations

# Đọc dữ liệu
df = pd.read_csv('synthetic_train.csv')

# 1. Tiền xử lý
def load_comprehensive_stopwords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Đồng bộ hóa định dạng khoảng trắng thành gạch dưới
            standard_sw = [line.strip().replace(" ", "_") for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy {filepath}")
        standard_sw = []
   
    # Loại bỏ nhiễu thực thể đặc thù (Domain-specific Noise)
    domain_noise = [
        "giảng_viên", "sinh_viên", "môn_học", "nhà_trường", "đội_ngũ", "phòng",
        "thiết_bị", "hệ_thống", "học_phần", "vấn_đề", "nội_dung", "phương_pháp",
        "thầy", "cô", "trường", "lớp", "việc", "cho", "của", "các", "những"
    ]
    return set(standard_sw + domain_noise)

stop_words = load_comprehensive_stopwords("vietnamese-stopwords.txt")

print("[1/4] Đang tiền xử lý dữ liệu...")
clean_data = []
for text in df['sentence']:
    tokens = word_tokenize(str(text), format="text").lower()
    filtered = [w for w in tokens.split() if w not in stop_words]
    clean_data.append(" ".join(filtered))

# 2. Vector hóa TF-IDF
print("[2/4] Đang vector hóa TF-IDF (N-gram 1,2)...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
X = vectorizer.fit_transform(clean_data)

# 3. Phân cụm K-Means
def find_k_optimized(X, max_k=3):
    wcss = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=5)
        km.fit(X)
        wcss.append(km.inertia_)
   
    if max_k <= 3: return max_k
   
    # Tính điểm khuỷu tay bằng khoảng cách hình học
    p1 = np.array([k_range[0], wcss[0]])
    p2 = np.array([k_range[-1], wcss[-1]])
    distances = []
    for i in range(len(k_range)):
        p0 = np.array([k_range[i], wcss[i]])
        d = np.abs(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)
        distances.append(d)
    return k_range[np.argmax(distances)]

print("[3/4] Đang tối ưu hóa số cụm và phân loại...")
k_opt = find_k_optimized(X, max_k=3)
model = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
df['Cluster'] = model.fit_predict(X)


print(f"\nHệ thống đã chốt số cụm: K = {k_opt}")

# 4. Đánh giá và trực quan hóa
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

print("\n[4/4] Đang vẽ biểu đồ PCA 2D...")
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

plt.title("KMeans + TF-IDF")
plt.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
k_values = list(range(1, 21))

cross_val_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy') 
    cross_val_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_values, cross_val_scores, marker='o', linestyle='-', color='b')
plt.title('Cross validasi untuk setiap nilai k')
plt.xlabel('nilai k')
plt.ylabel('Akurasi Cross-Validation')
plt.grid(True)
plt.show()

k_terbaik = k_values[cross_val_scores.index(max(cross_val_scores))]
print(f'nilai k terbaik: {k_terbaik}')


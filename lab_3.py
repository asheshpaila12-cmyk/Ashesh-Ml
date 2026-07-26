import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import minkowski

#A1
def label_encoding(df, column_name):
    unique_values = set(df[column_name])
    encoding_map = {value: idx for idx, value in enumerate(unique_values)}
    df[column_name] = df[column_name].map(encoding_map)
    return df, encoding_map

def one_hot_encoding(df, column_name):
    unique_values = df[column_name].unique()
    for value in unique_values:
        df[column_name + '_' + str(value)] = (df[column_name] == value).astype(int)
    df = df.drop(column_name, axis=1)
    return df

#A2
df = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df, _ = label_encoding(df, col)

feature_dimensionality = df.shape[1]
print("A2 - Feature Dimensionality after encoding:", feature_dimensionality)

#A3
df_onehot = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
for col in categorical_cols:
    df_onehot = one_hot_encoding(df_onehot, col)
print("A3 - Feature Dimensionality after one-hot encoding:", df_onehot.shape[1])

#A4
def minkowski_distance(u, v, p):
    sum_val = 0
    for i in range(len(u)):
        sum_val += abs(u[i] - v[i]) ** p
    return sum_val ** (1/p)

#A5
u = df.iloc[0][['MntSweetProducts', 'MntGoldProds']].values
v = df.iloc[1][['MntSweetProducts', 'MntGoldProds']].values

p_values = list(range(1, 11))
distances = [minkowski_distance(u, v, p) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, distances, marker='o', color='blue', linewidth=2, markersize=8)
plt.title("Minkowski Distance vs p", fontsize=14, fontweight='bold')
plt.xlabel("p", fontsize=12)
plt.ylabel("Minkowski Distance", fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(p_values)
plt.show()

#A6
distances_custom = [minkowski_distance(u, v, p) for p in p_values]
distances_scipy = [minkowski(u, v, p) for p in p_values]

print("\nA6 - Distance Comparison:")
print("p\tCustom\t\tScipy\t\tDifference")
for p, d_custom, d_scipy in zip(p_values, distances_custom, distances_scipy):
    print(f"{p}\t{d_custom:.6f}\t{d_scipy:.6f}\t{abs(d_custom - d_scipy):.10f}")

#A7
def dot_product(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

def euclidean_norm(vector):
    sum_sq = 0
    for val in vector:
        sum_sq += val ** 2
    return sum_sq ** 0.5

vector_a = np.array([2, 4, 6, 8, 10])
vector_b = np.array([1, 3, 5, 7, 9])

dot_custom = dot_product(vector_a, vector_b)
dot_numpy = np.dot(vector_a, vector_b)
norm_custom = euclidean_norm(vector_a)
norm_numpy = np.linalg.norm(vector_a)

print("\nA7 - Dot Product Comparison:")
print(f"Custom function: {dot_custom}")
print(f"NumPy function: {dot_numpy}")
print(f"Difference: {abs(dot_custom - dot_numpy)}")

print("\nA7 - Euclidean Norm Comparison:")
print(f"Custom function: {norm_custom:.6f}")
print(f"NumPy function: {norm_numpy:.6f}")
print(f"Difference: {abs(norm_custom - norm_numpy):.10f}")

#A8
def calculate_mean(data):
    if isinstance(data, pd.DataFrame):
        means = []
        for col in data.columns:
            col_sum = 0
            count = 0
            for val in data[col]:
                if pd.notna(val):
                    col_sum += val
                    count += 1
            means.append(col_sum / count if count > 0 else 0)
        return np.array(means)
    else:
        total = 0
        count = 0
        for val in data:
            if pd.notna(val):
                total += val
                count += 1
        return total / count if count > 0 else 0

def calculate_variance(data, mean_val=None):
    if mean_val is None:
        mean_val = calculate_mean(data)
    
    if isinstance(data, pd.DataFrame):
        variances = []
        for idx, col in enumerate(data.columns):
            sum_sq_diff = 0
            count = 0
            for val in data[col]:
                if pd.notna(val):
                    sum_sq_diff += (val - mean_val[idx]) ** 2
                    count += 1
            variances.append(sum_sq_diff / count if count > 0 else 0)
        return np.array(variances)
    else:
        sum_sq_diff = 0
        count = 0
        for val in data:
            if pd.notna(val):
                sum_sq_diff += (val - mean_val) ** 2
                count += 1
        return sum_sq_diff / count if count > 0 else 0

def calculate_std_dev(data, variance_val=None):
    if variance_val is None:
        variance_val = calculate_variance(data)
    
    if isinstance(data, pd.DataFrame):
        return np.sqrt(variance_val)
    else:
        return variance_val ** 0.5

def calculate_stats(data):
    mean_val = calculate_mean(data)
    variance_val = calculate_variance(data, mean_val)
    std_val = calculate_std_dev(data, variance_val)
    return mean_val, variance_val, std_val

numeric_data = df.select_dtypes(include=[np.number])
mean_custom, var_custom, std_custom = calculate_stats(numeric_data)

print("\nA8 - Custom Statistics for first 5 features:")
for i, col in enumerate(numeric_data.columns[:5]):
    print(f"{col}: Mean={mean_custom[i]:.4f}, Var={var_custom[i]:.4f}, Std={std_custom[i]:.4f}")

#A9
mean_numpy = np.mean(numeric_data, axis=0)
std_numpy = np.std(numeric_data, axis=0)

print("\nA9 - Comparison for first 5 features:")
print("\nMean Comparison:")
for i, col in enumerate(numeric_data.columns[:5]):
    print(f"{col}: Custom={mean_custom[i]:.6f}, NumPy={mean_numpy[i]:.6f}, Diff={abs(mean_custom[i] - mean_numpy[i]):.10f}")

print("\nStandard Deviation Comparison:")
for i, col in enumerate(numeric_data.columns[:5]):
    print(f"{col}: Custom={std_custom[i]:.6f}, NumPy={std_numpy[i]:.6f}, Diff={abs(std_custom[i] - std_numpy[i]):.10f}")

#A10
selected_feature = numeric_data.columns[0]
feature_values = numeric_data[selected_feature].dropna().values

plt.figure(figsize=(10, 6))
hist_data, bins, patches = plt.hist(feature_values, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
plt.title(f"Histogram of {selected_feature}", fontsize=14, fontweight='bold')
plt.xlabel(selected_feature, fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, alpha=0.3)
plt.axvline(calculate_mean(feature_values), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {calculate_mean(feature_values):.2f}')
plt.legend()
plt.show()

feature_mean = calculate_mean(feature_values)
feature_var = calculate_variance(feature_values, feature_mean)

print(f"\nA10 - Statistics for {selected_feature}:")
print(f"Mean: {feature_mean:.4f}")
print(f"Variance: {feature_var:.4f}")
print(f"Standard Deviation: {feature_var ** 0.5:.4f}")
print(f"Number of data points: {len(feature_values)}")
print(f"Histogram bins: {len(bins)-1}")
print(f"Data range: {feature_values.min():.2f} to {feature_values.max():.2f}")

#A11
def kmeans(X, k, max_iters=100, tol=1e-4):
    np.random.seed(42)
    n_samples, n_features = X.shape
    
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for iteration in range(max_iters):
        distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                distances[i, j] = minkowski_distance(X[i], centroids[j], 2)
        
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros((k, n_features))
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = cluster_points.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        centroid_shift = 0
        for j in range(k):
            centroid_shift += minkowski_distance(centroids[j], new_centroids[j], 2)
        
        centroids = new_centroids
        
        if centroid_shift < tol:
            break
    
    return centroids, labels

X_sample = numeric_data.values[:200]
k = 3
centroids, labels = kmeans(X_sample, k)

print("\nA11 - K-means Clustering Results:")
print(f"Number of clusters: {k}")
print(f"Number of samples: {len(X_sample)}")
print(f"Number of features: {X_sample.shape[1]}")
print(f"Iterations completed: {len(np.unique(labels))}")
print("\nCluster centroids (first 5 features):")
for i in range(k):
    print(f"Cluster {i+1}: {centroids[i][:5]}")

print(f"\nCluster distribution:")
for i in range(k):
    print(f"Cluster {i+1}: {np.sum(labels == i)} samples")

print("\nAll tasks completed successfully!")
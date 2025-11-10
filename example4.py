import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load Dataset
def load_dataset(path):
    try:
        data = pd.read_csv(path)
        print("Dataset loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Step 2: Preprocess Dataset (drop non-numeric columns)
def preprocess_data(data):
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    return scaled_data

# Step 3: Determine Optimal Number of Clusters using Elbow Method
def plot_elbow_curve(data):
    wcss = []  # Within-cluster sum of squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# Step 4: Apply KMeans Clustering
def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

# Main Function
def main():
    file_path = input("Enter path to the CSV dataset: ")
    dataset = load_dataset(file_path)

    if dataset is None:
        return

    print("\nFirst 5 rows of the dataset:")
    print(dataset.head())

    scaled_data = preprocess_data(dataset)

    print("\nPlotting Elbow Curve to determine optimal number of clusters...")
    plot_elbow_curve(scaled_data)

    k = int(input("Enter the number of clusters (k): "))
    clusters = apply_kmeans(scaled_data, k)

    dataset['Cluster'] = clusters
    print("\nClustered Dataset (with 'Cluster' column added):")
    print(dataset.head())

    # Optional: Save the clustered dataset to a new CSV file
    dataset.to_csv("clustered_output.csv", index=False)
    print("\nClustered dataset saved as 'clustered_output.csv'.")

if __name__ == "__main__":
    main()

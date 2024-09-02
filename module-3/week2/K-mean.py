from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt




class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k  # Số lượng centroids (cụm)
        self.max_iters = max_iters  # Số lần lặp tối đa
        self.centroids = None  # Centroids ban đầu
        self.clusters = None  # Nhóm các điểm dữ liệu

    # Khởi tạo ngẫu nhiên các centroids
    def initialize_centroids(self, data):
        np.random.seed(42)  # Đặt seed để đảm bảo tính tái hiện của kết quả
        self.centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
    
    # Tính khoảng cách Euclid giữa hai điểm dữ liệu
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    # Gán cụm cho từng điểm dữ liệu dựa trên khoảng cách đến centroids
    def assign_clusters(self, data):
        distances = np.array([[self.euclidean_distance(x, centroid) for centroid in self.centroids] for x in data])
        return np.argmin(distances, axis=1)  # Trả về chỉ số của cụm gần nhất
    
    # Cập nhật centroids bằng cách tính trung bình của các điểm dữ liệu trong cụm
    def update_centroids(self, data):
        return np.array([data[self.clusters == i].mean(axis=0) for i in range(self.k)])
    
    # Phương thức chính để chạy thuật toán K-Means
    def fit(self, data):
        self.initialize_centroids(data)  # Khởi tạo centroids
        for i in range(self.max_iters):
            # Gán cụm cho các điểm dữ liệu
            self.clusters = self.assign_clusters(data)
            
            # Hiển thị các cụm hiện tại
            self.plot_clusters(data, i)
            
            # Cập nhật centroids mới
            new_centroids = self.update_centroids(data)
            
            # Kiểm tra điều kiện dừng: nếu centroids không thay đổi, dừng thuật toán
            if np.all(self.centroids == new_centroids):
                break
            
            # Cập nhật centroids
            self.centroids = new_centroids
        
        # Hiển thị kết quả cuối cùng
        self.plot_final_clusters(data)
    
    # Hiển thị các cụm ở từng bước lặp
    def plot_clusters(self, data, iteration):
        plt.scatter(data[:, 0], data[:, 1], c=self.clusters, cmap='viridis', marker='o', alpha=0.6)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=300, c='red', marker='x')
        plt.title(f"Iteration {iteration + 1}")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    
    # Hiển thị cụm và centroids cuối cùng sau khi thuật toán kết thúc
    def plot_final_clusters(self, data):
        plt.scatter(data[:, 0], data[:, 1], c=self.clusters, cmap='viridis', marker='o', alpha=0.6)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=300, c='red', marker='x')
        plt.title("Final Clusters and Centroids")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Dữ liệu mẫu để kiểm tra
    iris_dataset = load_iris()
    data = iris_dataset.data
    data = iris_dataset.data[:, :2]
    kmeans = KMeans(k=2)
    kmeans.fit(data)
    kmeans = KMeans(k=3)
    kmeans.fit(data)
    kmeans = KMeans(k=4)
    kmeans.fit(data)

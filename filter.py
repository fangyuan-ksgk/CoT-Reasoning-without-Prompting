# DataPoint Embedding, Filtering and Preprocessing
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import PIL
from PIL import Image


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_beautiful_embeddings(embeddings, title:str="Embedding PCA", draw=True):
    # Improve the aesthetics of the PCA scatter plot
    pca_result = PCA(n_components=2).fit_transform(embeddings)
    x = pca_result[:, 0]
    y = pca_result[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and an axes object
    scatter = ax.scatter(x, y, c=np.arange(len(embeddings)), cmap='viridis', alpha=0.6, edgecolors='w', s=80)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel('Principal Component 1', fontsize=14)
    ax.set_ylabel('Principal Component 2', fontsize=14)
    fig.colorbar(scatter, ax=ax, label='Data Point Index')
    ax.grid(True)  # Add grid for better readability
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Annotate the points with their index
    for i, txt in enumerate(range(len(embeddings))):
        ax.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

    if not draw:
        # print("Trying to close the figure")
        plt.close(fig)

    return fig2img(fig)

# def plot_beautiful_embeddings(embeddings, title:str="Embedding PCA"):
#     # Improve the aesthetics of the PCA scatter plot
#     pca_result = PCA(n_components=2).fit_transform(embeddings)
#     x = pca_result[:, 0]
#     y = pca_result[:, 1]

#     plt.figure(figsize=(10, 8))  # Set the figure size
#     scatter = plt.scatter(x, y, c=np.arange(len(embeddings)), cmap='viridis', alpha=0.6, edgecolors='w', s=80)
#     plt.title(title, fontsize=18)
#     plt.xlabel('Principal Component 1', fontsize=14)
#     plt.ylabel('Principal Component 2', fontsize=14)
#     plt.colorbar(scatter, label='Data Point Index')
#     plt.grid(True)  # Add grid for better readability
#     plt.axhline(0, color='black', linewidth=0.5)
#     plt.axvline(0, color='black', linewidth=0.5)

#     # Annotate the points with their index
#     for i, txt in enumerate(range(len(embeddings))):
#         plt.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')

#     plt.show()

def plot_embeddings(embeddings, title:str="Embedding PCA"):
    n = len(embeddings)
    scores = (embeddings @ embeddings.T) * 100
    pca = PCA(n_components=2)
    scores = scores
    pca.fit(scores)
    pca.transform(scores)
    plt.scatter(pca.components_[0], pca.components_[1], c=range(n))
    for i, txt in enumerate(range(n)):
        plt.annotate(i, (pca.components_[0][i], pca.components_[1][i]))
    plt.show()

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



class ClusterFilter:
    def __init__(self, ratio: float = 0.8):
        self.filter_ratio = ratio
        self.kmeans = None
        self.k = None

    def fit(self, data):
        if (self.k is not None and len(data) < self.k) or (self.k is None) or (self.kmeans is None):
            self.k = int(len(data) * self.filter_ratio)
            self.kmeans = KMeans(n_clusters=self.k, random_state=0)
        self.kmeans.fit(data)
        self.cluster_centers = self.kmeans.cluster_centers_
        self.inertia = self.kmeans.inertia_
        self.n_iter = self.kmeans.n_iter_
        self.labels = self.kmeans.labels_

    def get_avg_distance(self, data, threshold_ratio: float = None):
        self.fit(data)
        if threshold_ratio is not None:
            self.threshold_ratio = threshold_ratio
        cum_dist = 0
        cum_n = 0
        for i, label in enumerate(self.labels):
            center = self.cluster_centers[label]
            distance = np.linalg.norm(data[i] - center)
            cum_dist += distance
            cum_n += 1
        avg_dist = (cum_dist / cum_n)
        threshold_distance = avg_dist * 0.8
        return threshold_distance
    
    def pick_k_per_cluster(self, data, k: int = 1):
        """
        Pick k data points per cluster
        Return index of filtered data, as well as the filtered data
        """
        self.fit(data)
        filtered_data = []
        filtered_indices = []
        filtered_data_per_labels = {}
        for i, label in enumerate(self.labels):
            if label not in filtered_data_per_labels:
                filtered_data_per_labels[label] = 1
                filtered_data.append(data[i])
                filtered_indices.append(i)
            elif filtered_data_per_labels[label] < k:
                filtered_data_per_labels[label] += 1
                filtered_data.append(data[i])
                filtered_indices.append(i)
        return np.array(filtered_indices), np.array(filtered_data)

    def filter(self, data, threshold_ratio: float = None):
        if threshold_ratio is not None:
            self.threshold_ratio = threshold_ratio
        threshold_distance = self.get_avg_distance(data)
        filtered_data = []
        for i, label in enumerate(self.labels):

            center = self.cluster_centers[label]
            distance = np.linalg.norm(data[i] - center)
            if distance < threshold_distance:
                filtered_data.append(data[i])
        return np.array(filtered_data)    
    
    
def present_filtering_result(embeddings, filtered_embeddings):
    pre_filter_img = plot_beautiful_embeddings(embeddings, draw=False)
    post_filter_img = plot_beautiful_embeddings(filtered_embeddings, draw=False)

    # Concatenate the images side by side
    combined_img = Image.new('RGB', (pre_filter_img.width + post_filter_img.width, pre_filter_img.height))
    combined_img.paste(pre_filter_img, (0, 0))
    combined_img.paste(post_filter_img, (pre_filter_img.width, 0))

    # Display the combined image
    combined_img.show()
    return combined_img
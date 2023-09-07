import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import pandas as pd
import math
import time
from sklearn.neighbors import KNeighborsClassifier
import sys

# Get the filename from the command-line argument
filename = sys.argv[1]

# Read and process the contents of the file
try:
    dataset=np.load(filename,allow_pickle=True)
except FileNotFoundError:
    print("File '{}' not found.".format(filename))



# Extract the label column (assuming it's the fourth column, index 3)
labels = dataset[:, 3]

# Count the frequency of each label
label_counts = np.unique(labels, return_counts=True)

# Extract label names and their corresponding frequencies
label_names = label_counts[0]
label_frequencies = label_counts[1]

length=len(label_names)
x_resnet=dataset[:,1]
x_vit=dataset[:,2]
y=dataset[:,3]

class KNN:
    def __init__(self,encoder,distance_metric,k):
        self.encoder=encoder
        self.distance_metric=distance_metric
        self.k=k
        self.y=y
        
    def set_encoder(self,encoder):
        self.encoder=encoder
        if(self.encoder=="VIT"):
            self.x=x_vit
        elif(self.encoder=="ResNet"):
            self.x=x_resnet
    
    def set_k(self,k):
        self.k=k
    def set_distance_metric(self,distance_metric):
        self.distance_metric=distance_metric

    def euclidean_distance(self,v1,v2):
        return np.linalg.norm(v1 - v2)
    def manhattan_distance(self,v1,v2):
        return np.sum(np.abs(v1 - v2))
    def cosine_distance(self,v1,v2):
        v1 = v1.reshape(-1)
        v2 = v2.reshape(-1)
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        similarity = dot_product / (norm1 * norm2)
        return 1 - similarity  # Convert similarity to distance

    def predict (self,v):
        distances = []

        for data_point, label in zip(self.data, self.labels):
            if self.distance_metric == 'Euclidean':
                dist = self.euclidean_distance(v, data_point)
            elif self.distance_metric == 'Manhattan':
                dist = self.manhattan_distance(v, data_point)
            elif self.distance_metric == 'Cosine':
                dist = self.cosine_distance(v, data_point)
            else:
                raise ValueError("Invalid distance metric")

            distances.append((dist, label))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]

        neighbor_labels = [neighbor[1] for neighbor in neighbors]
        most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)

        return most_common_label
    
    def split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.1, random_state=42)
        self.x_train=x_train
        self.x_test=x_test
        self.y_train=y_train
        self.y_test=y_test
        self.data=self.x_train
        self.labels=self.y_train
    
    def predict_all(self):
        predicted_labels=[]
        true_labels=[]
        # correct = 0
        for i in range(len(self.x_test)):
            predicted_labels.append(self.predict(self.x_test[i]))
            true_labels.append(self.y_test[i])

        self.f1 = f1_score(true_labels, predicted_labels, average='weighted')
        self.accuracy = accuracy_score(true_labels, predicted_labels)
        self.precision = precision_score(true_labels, predicted_labels, average='weighted',zero_division=1)
        self.recall = recall_score(true_labels, predicted_labels, average='weighted',zero_division=1)

encoder_type_list=["VIT","ResNet"]
distance_metric_list=["Euclidean","Manhattan","Cosine"]
k_value=4 
knn=KNN(k=k_value,encoder=encoder_type_list[0],distance_metric=distance_metric_list[0])
knn.set_encoder(encoder_type_list[0])
knn.split()
knn.predict_all()

print("F1-score:", knn.f1)
print("Accuracy:", knn.accuracy)
print("Precision:", knn.precision)
print("Recall:", knn.recall)
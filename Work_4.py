# -*- summary -*-
"""
The code performs hierarchical clustering on a dataset (covid19_stocks.csv).
It normalizes the data using Min-Max Scaling before clustering.
Different linkage methods (single, complete, average, ward) are tested for clustering.
For each method, scatter plots visualize cluster assignments.
Dendrograms illustrate hierarchical clustering structure.
The approach helps analyze relationships between stock data points.
"""
# -*- summary -*-

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

clustering_linkages = ["single","complete","average","ward"]
dendrogram_linkages = ["single","complete","average","centroid"]

data = np.genfromtxt("covid19_stocks.csv", delimiter = ",", skip_header = True)

plt.figure()
plt.scatter(data[:,0],data[:,1])
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("DataSet")
plt.show()

scaler = MinMaxScaler()

min_max_data = scaler.fit_transform(data)
dendrogram_linkages_index = 0

for linkage in clustering_linkages:
    
        plt.figure()
        fig, axs = plt.subplots(2, figsize=(25, 12)) 
        fig.tight_layout(h_pad=2)

        fig.suptitle(linkage  + " linkage")


        model = AgglomerativeClustering(n_clusters = 2, linkage = linkage)
        model.fit(min_max_data)
        labels = model.labels_
        
        axs[0].scatter(min_max_data[:,0],min_max_data[:, 1],c = labels, cmap = "rainbow")
        plt.xlabel("Feature1")
        plt.ylabel("Feature2")
        plt.title(linkage+"-linkage")
        

        axs[1] = sch.dendrogram(sch.linkage(data, method = dendrogram_linkages[dendrogram_linkages_index]), labels = labels)

        # Givng titles for dendrogram, x axis & y axis:
        plt.title('Dendrogram - ' + dendrogram_linkages[dendrogram_linkages_index])
        plt.xlabel('Clusters')
        plt.ylabel('Euclidean distances')
        plt.show()
  
        dendrogram_linkages_index+=1
    
# Plot the hierarchical clustering as a dendrogram

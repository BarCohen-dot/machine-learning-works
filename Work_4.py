# -*- summary -*-
"""
Example Of Hierarchical Clustering implementation . In this code, you'll see:
    1. Implementation of Hierarchical Clustering covid19 stock dataset
    2. Plotting the dendrogram
    3. Plotting a scatter plot for the data after assigning each sample to a cluster

Notes:
    1. In order to run this script, the dataset must be supplied.
       If the dataset is in the same directory of this script, then only the name of the dataset 
       must be supplied otherwise, the whole dataset's(In this case csv file) path must be supplied.
    
    2. Since we're running the HC for two columns, having different value ranges,
       therefore we need to scale our data.
      
    3. In this script, we're using the following libraries: 
        A. sklearn library for clustering methods
        B. scipy for plotting the dendrogram
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

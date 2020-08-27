# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:33:57 2020

@author: VIDHI
"""

#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import time
from numpy import nan
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from scipy import stats

#%%
#Importing Dataset: Dow Jones Index
data1 = pd.read_csv(r"C:\Users\VIDHI\Desktop\Dow Jones Index.csv")

#%%
#Exploratory Data Analysis
print(data1.shape)
data1.info()
data1.describe()
data1.head()

#%%
#feature data and removing date and stock
data1 = data1.drop('date', axis=1)
data1 = data1.drop('stock', axis=1)

#%%
#to check for correlations
sns.pairplot(data1)
#%%
#correlations
data1.corr().abs()

#%%
#heatmap
cmap = sns.diverging_palette(h_neg=10,h_pos=240,as_cmap = True)
cmao = sns.heatmap(data1.corr(), center = 0, cmap = cmap, annot = True, fmt = '.2f')

#%%
data11 = pd.DataFrame(data1)
#%%
#open high low previous weeks volume, next week open must be removed

data11 = data11.drop(data11.columns[[1,2,3,8,9]], axis = 1)

#%%
#Identifying outliers
k = np.abs(stats.zscore(data11))
print(k)

#setting threshold
threshold = 3
print(np.where(k> 3))
#%%
#removing outliers
data11 =data11[(k<3).all(axis=1)]
print(data11.shape)
#%%
#assessing variance
data11.var()
#%%
#since features have different variances, data needs to be scaled
ss = StandardScaler()
scaled = ss.fit_transform(data11)
scaled_ds = pd.DataFrame(scaled)

#%%
#visualizing high dimensional data to get insights
# Create a TSNE instance: model
model_tsne = TSNE(learning_rate=500)

# Apply fit_transform to samples: tsne_features
tsne_features = model_tsne.fit_transform(scaled_ds)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys)
plt.show()
#%%
#Feature selection : PCA
fmodel = PCA(n_components=5)
pca_features = fmodel.fit_transform(scaled_ds)

#%%
#creating KMeans model and measuring time
model = KMeans(n_clusters = 2)
model.fit(pca_features)
labels = model.predict(pca_features)
#%%
#measuring SSE
print(model.inertia_)
#%%
#To select best k value
ks = range(1, 20)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(pca_features)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#%%
#measuring silhouette score
silhouette_score(pca_features, labels = labels)
samplevalues= silhouette_samples(pca_features,labels)
#%%
#Silhouette plot
range_n_clusters = [2, 3, 4, 5, 6, 7]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(pca_features) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans()
    cluster_labels = clusterer.fit_predict(pca_features)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(pca_features, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(pca_features, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(pca_features[:, 0], pca_features[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    

    

    plt.suptitle(("Silhouette analysis for Agglomerative clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()


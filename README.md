# TimeSeriesAnalysis

Projekt for CDS TimeSeries Analysis

## Clustering Time series

For this clustering task, we select the feature 'Store', 'Date' and 'Weekly_Sales'. In order to group the series that share similar sales patterns (trend, seasonality, peak size).

The according Jupiter notebook can be found [here](./clustering/Clustering.ipynb).

### Workflow:

1. Using 'pivot_table' to reorganize data where each row represent each 'Store' and columns are the date records.
2. Scaling the 'pivot_table' using StandardScaler() because the sales are in much different range.
3. Clustering using K-Means through the function 'get_kmeans_results(data, max_clusters=10, metric='euclidean', seed=23)'. For the clustering metric, we can use DTW instead of euclidean method. Additionaly, we can maybe experiement with different number seed.
4. Visualisation the cluster central points through the function 'plot_clusters'

### Result

1. At the beginning, the Silhouette Score suggest 2 clusters but it doesnt a good fit to our data.

2. Experiment with Downsizing using TSNE, the Silhouette Score suggest 10 clusters. But we also experiement with 16 and it seems better: 
![16 clusters using metric of DTW](TSNE_16_clusters.png)

3. Experiment with Downsizing using MultiDimensional Scaling (MDS)
![6 clusters using metric of DTW ](MDS_6_clusters.png)



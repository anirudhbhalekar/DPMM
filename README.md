# Gaussian Mixture Models
Implementation of EM and DPMM (Dirichlet Process Mixture Model) algorithms for multivariate cluster analysis. Streamlined for Supervised-Unsupervised learning, made to work with csv files with labelled data

# How to use
The main class to be interacted with for general purpose clustering is in `Cluster.py`. Specifically in the `Clusterer()` class. The steps used to typically load a dataset, train the model, test the model, generate test visualisations, and store it are detailed below. 

Ensure your csv file is row-wise (i.e. label and corresponding dataset belong in the same row). Column-wise loading will be added soon. The following is what would typically written to train, test, and store a model based on a datset.

```
filename = "your_data.csv"

clusterer = Clusterer()
clusterer.load_csv(f"{filename}.csv", label_col = 0, d_start_col = 2, d_end_col = 45)
clusterer.DPMM_train(train_size=600, n_components=10, num_categories=-1, degradation=0)
clusterer.DPMM_test()
clusterer.heat_map()
clusterer.store_dpmm()
```


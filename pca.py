def get_reduced_data(data , X_test , numoffeatures):
    from sklearn.decomposition import PCA
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    pca = PCA(n_components=numoffeatures)  # Keep top 100 features
    X_pca = pca.fit_transform(data)
    X_pca2 = pca.transform(X_test)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Total explained variance with {numoffeatures} features: {explained_variance:.2f}%")
    reduced_data = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(numoffeatures)]).values
    reduced_test = pd.DataFrame(X_pca2,  columns=[f'PC{i+1}' for i in range(numoffeatures)]).values
    return reduced_data , reduced_test
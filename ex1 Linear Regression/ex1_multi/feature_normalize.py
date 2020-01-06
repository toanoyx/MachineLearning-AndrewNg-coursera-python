def feature_normalize(X):
    mu = X.mean(axis=0)  # 每一列/特征的均值
    sigma = X.std(axis=0)  # 每一列/特征的标准差
    X_norm = (X - mu) / sigma  # 广播机制

    return X_norm, mu, sigma

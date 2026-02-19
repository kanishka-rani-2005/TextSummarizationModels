import numpy as np

def calculate_topsis(decision_matrix, weights, impacts):
    X = np.array(decision_matrix, dtype=float)
    weights = np.array(weights, dtype=float)

    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1  # avoid division by zero
    X_norm = X / norm

    X_weighted = X_norm * weights

    ideal_best = []
    ideal_worst = []

    for j in range(X.shape[1]):
        if impacts[j] == '+':
            ideal_best.append(X_weighted[:, j].max())
            ideal_worst.append(X_weighted[:, j].min())
        else:
            ideal_best.append(X_weighted[:, j].min())
            ideal_worst.append(X_weighted[:, j].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    dist_best = np.sqrt(((X_weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((X_weighted - ideal_worst) ** 2).sum(axis=1))

    denom = dist_best + dist_worst
    denom[denom == 0] = 1e-10

    score = dist_worst / denom
    return score

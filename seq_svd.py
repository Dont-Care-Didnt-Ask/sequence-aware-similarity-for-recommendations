import numpy as np
import ilupp

import scipy
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix
from scipy.sparse import eye as speye

from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import generate_interactions_matrix, generate_weights
from evaluation import downvote_seen_items, topn_recommendations, model_evaluate

from itertools import product

def jaccard_similarity(matrix_A, matrix_B=None):
    if matrix_B is None:
        matrix_B = matrix_A

    similarity = lil_matrix((matrix_B.shape[0], matrix_A.shape[0]))
    for u in range(matrix_A.shape[0]):
        repeated_row_matrix = csr_matrix(np.ones([matrix_B.shape[0], 1])) * matrix_B[u]
        numerator = repeated_row_matrix.minimum(matrix_B).sum(axis=1)
        denominator = repeated_row_matrix.maximum(matrix_B).sum(axis=1) + 1e-3
        similarity[:, u] = numerator / denominator

    return csr_matrix(similarity.T)


def build_seq_svd_model(config, training, data_description):
    training_with_time_ranks = generate_weights(training, data_description, config["power"])

    weight_description = data_description.copy()
    weight_description["feedback"] = "weights"
    time_interactions = generate_interactions_matrix(training_with_time_ranks, weight_description)

    if config["similarity_type"] == "jaccard":
        S = jaccard_similarity(time_interactions)
    elif config["similarity_type"] == "cosine":
        S = csr_matrix(cosine_similarity(time_interactions, time_interactions))

    L = ilupp.icholt(S, add_fill_in=0, threshold=config["incomplete_cholesky_threshold"])

    alpha = config["sequence_similarity_coef"]
    L = alpha * L + (1 - alpha) * speye(S.shape[0])

    matrix = generate_interactions_matrix(training, data_description).astype(np.float32)
    weights = np.power(matrix.getnnz(axis=0) + 1, 0.5 * (config["scaling"] - 1))
    
    matrix = L.T.dot(matrix.dot(scipy.sparse.diags(weights)))

    _, _, vt = svds(matrix, k=config["rank"], return_singular_vectors="vh")
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors


def svd_scoring(params, data, data_description):
    item_factors = params
    test_matrix = generate_interactions_matrix(data, data_description, rebase_users=True)
    scores = test_matrix.dot(item_factors) @ item_factors.T
    downvote_seen_items(scores, data, data_description)
    return scores


def full_seq_svd_gridsearch(ranks, sequence_similarity_coefs, powers, thresholds, similarity_types, scalings,
        training, testset, holdout, data_description, topn):
    max_rank = max(ranks)
    config = {"rank": max_rank}
    results = []

    grid = product(sequence_similarity_coefs, powers, thresholds, similarity_types, scalings)

    for sequence_similarity_coef, power, threshold, sim_type, scaling in grid:
        print(f"Sequence similarity coef: {sequence_similarity_coef}, power: {power}, "
              f"threshold: {threshold}, similarity type: {sim_type}, scaling: {scaling}")
        
        config["incomplete_cholesky_threshold"] = threshold
        config["sequence_similarity_coef"] = sequence_similarity_coef
        config["similarity_type"] = sim_type
        config["power"] = power
        config["scaling"] = scaling

        item_factors = build_seq_svd_model(config, training, data_description)

        for rank in ranks:
            item_factors_trunc = item_factors[:, :rank]
            scores = svd_scoring(item_factors_trunc, testset, data_description)
            recs = topn_recommendations(scores, topn)
            metric = model_evaluate(recs, holdout, data_description, topn)

            results.append({
                "rank": rank,
                "incomplete_cholesky_threshold": threshold,
                "sequence_similarity_coef": sequence_similarity_coef,
                "similarity_type": sim_type,
                "power": power,
                "scaling": scaling,
                "metric": metric,
            })
        
    return results
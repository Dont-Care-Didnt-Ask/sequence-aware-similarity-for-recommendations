import numpy as np
from scipy.sparse.linalg import svds

from data_preprocessing import generate_interactions_matrix
from evaluation import downvote_seen_items, topn_recommendations, model_evaluate


def build_svd_model(config, training, data_description):
    matrix = generate_interactions_matrix(training, data_description).astype(np.float32)
    print("Interaction matrix shape:", matrix.shape)
    print("Matrix density:", matrix.getnnz() / np.prod(matrix.shape))
    _, _, vt = svds(matrix, k=config["rank"], return_singular_vectors="vh")
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors


def svd_scoring(item_factors, data, data_description):
    test_matrix = generate_interactions_matrix(data, data_description, rebase_users=True)
    scores = test_matrix.dot(item_factors) @ item_factors.T
    downvote_seen_items(scores, data, data_description)
    return scores


def svd_gridsearch(ranks, training, testset, holdout, data_description, topn):
    max_rank = max(ranks)
    config = {"rank": max_rank}
    item_factors = build_svd_model(config, training, data_description)
    results = []
    
    for rank in ranks:
        item_factors_trunc = item_factors[:, :rank]
        scores = svd_scoring(item_factors_trunc, testset, data_description)
        recs = topn_recommendations(scores, topn)
        metric = model_evaluate(recs, holdout, data_description, topn)

        results.append({
            "rank": rank, "metric": metric,
        })
    
    return results
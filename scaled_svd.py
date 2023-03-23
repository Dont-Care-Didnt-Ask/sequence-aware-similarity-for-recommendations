import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import svds

from data_preprocessing import generate_interactions_matrix
from evaluation import downvote_seen_items, topn_recommendations, model_evaluate
from pure_svd import svd_scoring


def build_scaled_svd_model(config, training, data_description):
    matrix = generate_interactions_matrix(training, data_description)
    weights = np.power(matrix.getnnz(axis=0) + 1, 0.5 * (config["scaling"] - 1))
    
    _, _, vt = svds(matrix.dot(sps.diags(weights)), k=config["rank"], return_singular_vectors="vh")
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors


def scaled_svd_gridsearch(ranks, scalings, training, testset, holdout, data_description, topn):
    max_rank = max(ranks)
    config = {"rank": max_rank}
    results = []
    
    for scaling in scalings:
        print("Scaling", scaling)

        config["scaling"] = scaling
        item_factors = build_scaled_svd_model(config, training, data_description)

        for rank in ranks:
            item_factors_trunc = item_factors[:, :rank]
            scores = svd_scoring(item_factors_trunc, testset, data_description)
            recs = topn_recommendations(scores, topn)
            metric = model_evaluate(recs, holdout, data_description, topn)
            
            results.append(
                {"rank": rank, "scaling": scaling, "metric": metric}
            )

    return results
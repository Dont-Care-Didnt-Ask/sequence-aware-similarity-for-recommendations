import numpy as np
import ilupp
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix
from scipy.sparse import eye as speye

from data_preprocessing import generate_interactions_matrix, generate_weights
from evaluation import downvote_seen_items, topn_recommendations, model_evaluate


def jaccard_similarity(matrix_A, matrix_B=None):
    if matrix_B is None:
        matrix_B = matrix_A

    similarity = lil_matrix((matrix_B.shape[0], matrix_A.shape[0]))
    for u in range(matrix_A.shape[0]):
        repeated_row_matrix = csr_matrix(np.ones([matrix_B.shape[0], 1])) * matrix_B[u]
        similarity[:, u] = (repeated_row_matrix.minimum(matrix_B).sum(axis=1) / repeated_row_matrix.maximum(matrix_B).sum(axis=1))

    return csr_matrix(similarity.T)


def build_jaccard_svd_model(config, training, data_description):
    train_jaccard_weights = generate_weights(training, data_description)

    weight_description = data_description.copy()
    weight_description["feedback"] = "timestamp"
    jaccard_interactions = generate_interactions_matrix(train_jaccard_weights, weight_description)
    #print("Computing jaccard similarity...")

    S = jaccard_similarity(jaccard_interactions)
    #print("Similarity matrix shape:", S.shape)
    #print("S density:", S.getnnz() / np.prod(S.shape))

    L = ilupp.icholt(S, add_fill_in=S.shape[0], threshold=0.05)

    L = config["jaccard_coef"] * L + (1 - config["jaccard_coef"]) * speye(S.shape[0])
    #print("L matrix shape:", L.shape)
    #print("L density:", L.getnnz() / np.prod(L.shape))

    matrix = generate_interactions_matrix(training, data_description).astype(np.float32)
    
    #print("Interaction matrix shape:", matrix.shape)
    #print("Matrix density:", matrix.getnnz() / np.prod(matrix.shape))

    matrix = L.T.dot(matrix)

    #print("L * Interaction matrix shape:", matrix.shape)
    #print("L * Matrix density:", matrix.getnnz() / np.prod(matrix.shape))


    _, _, vt = svds(matrix, k=config["rank"], return_singular_vectors="vh")
    item_factors = np.ascontiguousarray(vt[::-1, :].T)
    return item_factors


def svd_scoring(params, data, data_description):
    item_factors = params
    test_matrix = generate_interactions_matrix(data, data_description, rebase_users=True)
    scores = test_matrix.dot(item_factors) @ item_factors.T
    downvote_seen_items(scores, data, data_description)
    return scores


def jaccard_svd_gridsearch(ranks, jaccard_coefs, training, testset, holdout, data_description, topn):
    max_rank = max(ranks)
    config = {"rank": max_rank}
    results = []

    for jaccard_coef in jaccard_coefs:
        print(jaccard_coef)
        config["jaccard_coef"] = jaccard_coef
        item_factors = build_jaccard_svd_model(config, training, data_description)

        for rank in ranks:
            item_factors_trunc = item_factors[:, :rank]
            scores = svd_scoring(item_factors_trunc, testset, data_description)
            recs = topn_recommendations(scores, topn)
            metric = model_evaluate(recs, holdout, data_description, topn)

            results.append({
                "rank": rank, "metric": metric, "jaccard_coef": jaccard_coef,
            })
        
    return results
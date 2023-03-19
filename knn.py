import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from data_preprocessing import generate_interactions_matrix
from evaluation import downvote_seen_items, topn_recommendations, model_evaluate


def truncate_similarity(similarity, k):
    '''
    For every row in similarity matrix, pick at most k entities
    with the highest similarity scores. Disregard everything else.
    '''
    similarity = similarity.tocsr()
    inds = similarity.indices
    ptrs = similarity.indptr
    data = similarity.data
    new_ptrs = [0]
    new_inds = []
    new_data = []
    for i in range(len(ptrs)-1):
        start, stop = ptrs[i], ptrs[i+1]
        if start < stop:
            data_ = data[start:stop]
            topk = min(len(data_), k)
            idx = np.argpartition(data_, -topk)[-topk:]
            new_data.append(data_[idx])
            new_inds.append(inds[idx+start])
            new_ptrs.append(new_ptrs[-1]+len(idx))
        else:
            new_ptrs.append(new_ptrs[-1])
    new_data = np.concatenate(new_data)
    new_inds = np.concatenate(new_inds)
    truncated = csr_matrix(
        (new_data, new_inds, new_ptrs),
        shape=similarity.shape
    )
    return truncated 


def build_uknn_model(config, training, data_description):
    user_item_mat_train = generate_interactions_matrix(training, data_description).astype(np.float32)
    return user_item_mat_train


def uknn_scoring(user_item_mat_train, data, data_description, k=None):
    user_item_mat_test = generate_interactions_matrix(data, data_description, rebase_users=True)
    similarity = cosine_similarity(user_item_mat_test, user_item_mat_train, dense_output=False)

    if k is not None:
        similarity = truncate_similarity(similarity, k)

    scores = similarity.dot(user_item_mat_train).A
    downvote_seen_items(scores, data, data_description)
    return scores


def uknn_gridsearch(k_vals, training, testset, holdout, data_description, topn):
    user_item_mat_train = build_uknn_model(None, training, data_description)
    results = {}
    
    for k in k_vals:
        scores = uknn_scoring(user_item_mat_train, testset, data_description, k=k)
        recs = topn_recommendations(scores, topn)
        results[k] = model_evaluate(recs, holdout, data_description, topn)
    
    return results


def build_iknn_model(config, training, data_description):
    user_item_mat_train = generate_interactions_matrix(training, data_description).astype(np.float32)
    item_similarity = cosine_similarity(user_item_mat_train, dense_output=False)
    item_similarity.set_diag(0)
    return item_similarity


def iknn_scoring(item_similarity, data, data_description, k=None):
    user_item_mat_test = generate_interactions_matrix(data, data_description, rebase_users=True)

    if k is not None:
        item_similarity = truncate_similarity(item_similarity, k)

    scores = user_item_mat_test.dot(item_similarity.T).A
    downvote_seen_items(scores, data, data_description)
    return scores


def iknn_gridsearch(k_vals, training, testset, holdout, data_description, topn):
    item_similarity = build_iknn_model(None, training, data_description)
    results = {}
    
    for k in k_vals:
        scores = uknn_scoring(item_similarity, testset, data_description, k=k)
        recs = topn_recommendations(scores, topn)
        results[k] = model_evaluate(recs, holdout, data_description, topn)
    
    return results
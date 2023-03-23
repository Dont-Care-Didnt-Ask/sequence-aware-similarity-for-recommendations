import numpy as np
import pandas as pd

def downvote_seen_items(scores, data, data_description):
    assert isinstance(scores, np.ndarray), 'Scores must be a dense numpy array!'
    itemid = data_description['items']
    userid = data_description['users']
    # get indices of observed data, corresponding to scores array
    # we need to provide correct mapping of rows in scores array into
    # the corresponding user index (which is assumed to be sorted)
    row_idx, test_users = pd.factorize(data[userid], sort=True)
    assert len(test_users) == scores.shape[0]
    col_idx = data[itemid].values
    # downvote scores at the corresponding positions
    scores[row_idx, col_idx] = scores.min() - 1



def only_sample_items(scores, holdout, data_description, neg_sample_size):
    assert isinstance(scores, np.ndarray), 'Scores must be a dense numpy array!'
    itemid = data_description['items']
    col_idx_hol = holdout[itemid].values
    for i in range(scores.shape[0]):
        all = list(np.arange(scores.shape[1]))
        all.pop(col_idx_hol[i])
        ind = np.random.choice(all, neg_sample_size, replace = False)
        ind_del = list(set(all).symmetric_difference(set(ind)))
        scores[i, ind_del] = scores.min() - 1


def model_evaluate(recommended_items, holdout, holdout_description, topn):
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    assert recommended_items.shape[0] == len(holdout_items)
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
# HR calculation
    hr = np.mean(hits_mask.any(axis=1))
# MRR calculation
    n_test_users = recommended_items.shape[0]
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
# ncdg
    ndcg_per_user = 1 / np.log2(hit_rank + 1)
    ndcg = np.sum(ndcg_per_user) / n_test_users
# coverage calculation
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items

    #return ndcg, hr, mrr, cov
    return hr


def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]
import igraph as ig
import numpy as np
from scipy.stats import hmean

def is_dag(A):
    G = ig.Graph.Weighted_Adjacency(A.tolist())
    return G.is_dag()

def get_DAG(B):
    A = np.where(B!=0,1,0)
    return A

def get_CPDAG(B):
    C = get_DAG(B) 

    A = np.tril(C,-1)+np.tril(C.T,-1)
    
    C[np.where(A==2)]=-1
    C[np.where(A.T==2)]=0

    return C

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.
    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition
    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
                            If {0,1}, it must be a DAG
    Returns:
        fdr|(1-precision): (reverse + false positive) / prediction positive
        tpr|recall: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        f1: 2*((1-fdr)*tpr)/((1-fdr)+tpr)
        shd (plus components): undirected extra + undirected missing + reverse
        pred-und: number of undirected edges in B_est
        f-und: pred-und/nnz 
        nnz: prediction positive
        true_nnz: number of edges in B_true
        ae_nnz: |true_nnz-nnz|
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        # if not is_dag(B_est):
        #     raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    true_nnz = len(cond)
    cond_neg_size = 0.5 * d * (d - 1) - true_nnz
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(true_nnz, 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    f1 = hmean(np.array([1.-fdr, tpr]))

    return {'fdr|(1-precision)': fdr, 'tpr|recall': tpr, 'fpr': fpr, 'f1':f1,
            'shd-missing':len(missing_lower), 'shd-extra':len(extra_lower), 'shd-reverse':len(reverse), 'shd': shd,
            'pred-und':len(pred_und), 'f-und': len(pred_und)*1./max(pred_size, 1.), 
            'nnz': pred_size, 'true_nnz':true_nnz, 'ae_nnz':np.abs(pred_size-true_nnz)}

def _score(ordering):
    N=len(ordering)
    scores=dict()
    
    for i,o in enumerate(ordering):
        scores[o]=N-i
    return scores

def _map_to_score(ordering, score):
    return np.array([score[o] for o in ordering])

def _ndcg(scores, k=None):
    dcg = 0.
    idcg = 0.
    N=len(scores)
    
    if k==None: k=N
        
    for i,s in enumerate(scores[:k]):
        dcg+=s/np.log2(i+2)
        idcg+=(N-i)/np.log2(i+2)

    return dcg/idcg
"""Finite-menu robust split conformal; no model fitting occurs here."""
import numpy as np


def quantile(scores, alpha):
    if not 0 < alpha < 1: raise ValueError('alpha must be in (0,1)')
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores): raise ValueError('nonempty 1D scores required')
    if not np.isfinite(scores).all(): raise ValueError('scores must be finite')
    k = int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(np.partition(scores, k-1)[k-1]) if k <= len(scores) else float('inf')


def calibrate(probabilities, labels, alpha=.1, nonconformity=None):
    """probabilities: [context, image, class]; returns context thresholds.

    Envelope threshold is calibrated on the maximum true-label score, then
    used with the score of whichever context is actually deployed. It protects
    true-label inclusion simultaneously over the fixed menu under exchangeability.
    """
    m,n,_=probabilities.shape
    scores=1-probabilities if nonconformity is None else np.asarray(nonconformity)
    if scores.shape != probabilities.shape:raise ValueError('score shape must match probabilities')
    s=scores[:,np.arange(n),labels]
    nominal=quantile(s[0],alpha)
    from scipy.stats import rankdata
    ranks=rankdata(s,method='max',axis=1)
    qrank=quantile(ranks.max(0),alpha)
    rank_thresholds=np.sort(s,axis=1)[:,int(qrank)] if qrank < n else np.full(m,float('inf'))
    return {'nominal':np.full(m,nominal),
            'context':np.array([quantile(a,alpha) for a in s]),
            'bonferroni':np.array([quantile(a,alpha/m) for a in s]),
            'corrected_maxrank':rank_thresholds,
            'envelope':np.full(m,quantile(s.max(0),alpha))}


def measure(p, y, thresholds, nonconformity=None):
    scores=1-p if nonconformity is None else np.asarray(nonconformity)
    if scores.shape != p.shape:raise ValueError('score shape must match probabilities')
    included = scores <= np.asarray(thresholds)[:,None,None]
    truth = included[:,np.arange(len(y)),y]
    sizes = included.sum(-1)
    prediction = p.argmax(-1)
    return {'simultaneous_coverage':float(truth.all(0).mean()),
            'min_context_coverage':float(truth.mean(1).min()),
            'mean_context_coverage':float(truth.mean()),
            'mean_size':float(sizes.mean()),
            'intersection_size':float(included.all(0).sum(-1).mean()),
            'singleton_rate':float((sizes==1).mean()),
            'empty_rate':float((sizes==0).mean()),
            'class_flip_rate':float((prediction!=prediction[0]).any(0).mean()),
            'probability_range':float(np.ptp(p[:,np.arange(len(y)),y],axis=0).mean())}

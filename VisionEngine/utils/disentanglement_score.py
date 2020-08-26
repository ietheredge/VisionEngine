from sklearn.linear_model import Lasso
import numpy as np

def norm_entropy(p):
    '''p: probabilities '''
    n = p.shape[0]
    return - p.dot(np.log(p + 1e-12) / np.log(n + 1e-12))
def entropic_scores(r):
    '''r: relative importances '''
    r = np.abs(r)
    ps = r / np.sum(r, axis=0) # 'probabilities'
    hs = [1-norm_entropy(p) for p in ps.T]
    return hs
def mse(predicted, target):
    ''' mean square error '''
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted #(n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target #(n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0] #value not array
def rmse(predicted, target):
    ''' root mean square error '''
    return np.sqrt(mse(predicted, target))
def nrmse(predicted, target):
    ''' normalized root mean square error '''
    return rmse(predicted, target) / np.std(target)

def dissentanglement_score(z, inputs, h):
    R = []
    err = []
    for j in range(inputs.shape[1]):    
        model = Lasso(alpha=.02, max_iter=10000)
        model.fit(z[h][:,:], inputs[:, j])
        z_pred = model.predict(z[h][:,:])
        r = getattr(model, 'coef_')[:, None]
        R.append(np.abs(r))
        err.append(nrmse(z_pred, inputs[:, j]))
    R = np.hstack(R)

    # disentanglement
    disent_scores = entropic_scores(R.T)
    c_rel_importance = np.nansum(R,1) / np.nansum(R)
    disent_w_avg = np.nansum(np.array(disent_scores) * c_rel_importance)

    # completeness
    complete_scores = entropic_scores(R)
    complete_avg = np.nanmean(complete_scores)

    return disent_w_avg, complete_avg

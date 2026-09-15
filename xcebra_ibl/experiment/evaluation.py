"""Trial-level evaluation; fitting and model selection never use test targets."""
import numpy as np
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def split_trials(count, seed):
    if count < 10:
        raise ValueError('At least 10 retained trials are required')
    order = np.random.default_rng(seed).permutation(count)
    n = max(2, int(count * .2))
    return dict(train=np.sort(order[2*n:]), validation=np.sort(order[n:2*n]), test=np.sort(order[:n]))


def shuffle_trials(labels, trial_length, seed, groups=None):
    """One joint trial permutation preserves trajectories and label dependence."""
    count = len(next(iter(labels.values()))) // trial_length
    rng = np.random.default_rng(seed)
    permutation = np.arange(count)
    groups = np.zeros(count) if groups is None else np.asarray(groups)
    if len(groups) != count:
        raise ValueError("Permutation groups must match trials")
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        permutation[indices] = rng.permutation(indices)
    return {v: y.reshape(count, trial_length)[permutation].reshape(-1) for v, y in labels.items()}


def score(y, pred, discrete):
    if not discrete and np.var(y) < 1e-12:
        return None
    if discrete and np.unique(y).size < 2:
        return None
    value = balanced_accuracy_score(y, pred) if discrete else r2_score(y, pred)
    return float(value) if np.isfinite(value) else None


def interval(y, pred, trials, discrete, draws, seed, baseline=None, unit="held_out_trial"):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(trials == t) for t in np.unique(trials)]
    scores, differences = [], []
    for _ in range(draws if len(groups)>1 else 0):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        value = score(y[idx], pred[idx], discrete)
        if value is not None:
            scores.append(value)
            if baseline is not None:
                base_value = score(y[idx], baseline[idx], discrete)
                if base_value is not None:
                    differences.append(value-base_value)
    result = dict(score=score(y, pred, discrete), ci95=np.quantile(scores, [.025, .975]).tolist() if scores else None,
                  valid_bootstraps=len(scores), n_resampling_units=len(groups), unit=unit, metric='balanced_accuracy' if discrete else 'r2')
    if baseline is not None:
        base_score = score(y, baseline, discrete)
        result.update(baseline_score=base_score,
                      improvement=result['score']-base_score if result['score'] is not None and base_score is not None else None,
                      improvement_ci95=np.quantile(differences,[.025,.975]).tolist() if differences else None)
    return result



def decode(features, targets, discrete, include_knn=True, save_dir=None, prefix="decoder"):
    """Select decoder hyperparameters on validation trials only."""
    choices = []
    for alpha in (.1, 1., 10., 100.):
        estimator = (LogisticRegression(C=1/alpha, max_iter=2000, random_state=0)
                     if discrete else Ridge(alpha=alpha, solver="lsqr"))
        choices.append((f'linear_{alpha}', estimator))
    for k in ((5, 25, 100) if include_knn else ()):
        if k <= len(features['train']):
            choices.append((f'knn_{k}', KNeighborsClassifier(k) if discrete else KNeighborsRegressor(k)))
    if discrete and np.unique(targets['train']).size < 2:
        choices = [('constant', DummyClassifier(strategy='most_frequent'))]
    best = {}
    for name, estimator in choices:
        family = name.split('_')[0]
        model = make_pipeline(StandardScaler(), estimator)
        model.fit(features['train'], targets['train'])
        validation = score(targets['validation'], model.predict(features['validation']), discrete)
        if family not in best or (validation is not None and (best[family][0] is None or validation > best[family][0])):
            best[family] = validation, name, model
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for family, (_, _, fitted) in best.items():
            joblib.dump(fitted, save_dir / f"{prefix}_{family}.joblib")
    return {family: dict(validation_score=value, decoder=name,
                        prediction=model.predict(features['test']))
            for family, (value, name, model) in best.items()}


def encoding_baselines(session, splits, out):
    """Local time-resolved Ridge/RRR encoding; not the paper's global RRR fit."""
    X, Y = session['X_3d'], session['y_3d']
    output, coefficients, intercepts, test_predictions = {}, {}, {}, {}
    for rank in (None, 2, 4, 8):
        best = None
        for alpha in (.1, 1., 10., 100.):
            preds = {part: np.empty_like(Y[idx]) for part, idx in splits.items() if part != 'train'}
            fitted_coefficients, fitted_intercepts = [], []
            for t in range(session['T']):
                model = Ridge(alpha=alpha).fit(X[splits['train'], t], Y[splits['train'], t])
                center = model.intercept_
                projection = None
                if rank is not None:
                    fitted = model.predict(X[splits['train'], t]) - center
                    q, _ = np.linalg.qr(X[splits["train"], t] - X[splits["train"], t].mean(axis=0))
                    _, _, vt = np.linalg.svd(q.T @ fitted, full_matrices=False)
                    basis = vt[:min(rank, len(vt))].T
                    projection = basis
                fitted_intercepts.append(model.intercept_)
                coef = model.coef_.T
                fitted_coefficients.append(coef if projection is None else (coef @ projection) @ projection.T)
                for part in preds:
                    prediction = model.predict(X[splits[part], t])
                    preds[part][:, t] = prediction if projection is None else ((prediction-center) @ projection) @ projection.T + center
            value = r2_score(Y[splits['validation']].reshape(-1, session['N']), preds['validation'].reshape(-1, session['N']))
            if best is None or value > best[0]:
                best = value, alpha, preds['test'], np.asarray(fitted_coefficients), np.asarray(fitted_intercepts)
        name = 'ridge' if rank is None else f'rrr_rank{rank}'
        coefficients[name] = best[3]
        intercepts[name] = best[4]
        test_predictions[name] = best[2]
        output[name] = dict(
            validation_r2=float(best[0]), alpha=best[1],
            test_r2_per_neuron=r2_score(Y[splits['test']].reshape(-1, session['N']), best[2].reshape(-1, session['N']), multioutput='raw_values').tolist())
    np.savez_compressed(out / 'encoding_intercepts.npz', **intercepts)
    np.savez_compressed(out / 'encoding_coefficients.npz', **coefficients)
    np.savez_compressed(out / 'encoding_predictions.npz', truth=Y[splits['test']], **test_predictions)
    return output

#!/usr/bin/env python3
import argparse
import warnings
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm

warnings.filterwarnings("ignore", category=RuntimeWarning)


def clip_columns(df, cols, lo_q=1.0, hi_q=99.0):
    arr = df[cols].values.astype(float)
    lo, hi = np.percentile(arr, [lo_q, hi_q])
    clipped = np.clip(arr, lo, hi)
    df[cols] = clipped
    return df

def log_scale_df(df, cols):
    for c in cols:
        vals = df[c].astype(float).values
        sign = np.sign(vals)
        df[c] = sign * np.log1p(np.abs(vals))
    return df

def sliding_windows(seq, window_size, step):
    """Return list of windows (numpy arrays) from 1D or 2D seq."""
    res = []
    n = seq.shape[0]
    if n == 0:
        return res
    for start in range(0, max(1, n - window_size + 1), step):
        res.append(seq[start:start + window_size])
    # if not enough, still add short tail as one if nothing collected
    if len(res) == 0 and n > 0:
        res.append(seq)
    return res

# -----------------------
# Data preparation
# -----------------------
def prepare_sequences_from_csv(csv_path,
                               feature_cols=None,
                               label_col="pattern_type",
                               window_size=500,
                               window_step=200,
                               clip_percentiles=(1.0, 99.0),
                               do_log_scale=False):
    """
    Return:
      patterns -> list of sequences per pattern: dict pattern -> [np.array(seq_timesteps x D)]
      df (processed)
    """
    df = pd.read_csv(csv_path)
    if feature_cols is None:
        # heuristics: pick columns containing 'Delta_with' by default
        feature_cols = [c for c in df.columns if 'Delta_with' in c][:6]  # up-to-6 by default
        if not feature_cols:
            raise ValueError("No feature columns found; please specify feature_cols explicitly.")
    # fill NaNs
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # clip outliers jointly
    df = clip_columns(df, feature_cols, lo_q=clip_percentiles[0], hi_q=clip_percentiles[1])

    # optional log scaling
    if do_log_scale:
        df = log_scale_df(df, feature_cols)

    patterns = defaultdict(list)
    # group by label (pattern_type)
    for pat, group in df.groupby(label_col):
        arr = group[feature_cols].astype(float).values  # shape (T, D)
        # split into sliding windows to generate multiple sequences per pattern
        windows = sliding_windows(arr, window_size, window_step)
        patterns[pat].extend(windows)

    return patterns, df, feature_cols

# -----------------------
# Model training
# -----------------------
def train_discrete_hmms(pattern_seqs, n_bins=16, n_states=8, n_restarts=3):
    """
    pattern_seqs: dict pattern -> [np.array(seq_timesteps x D)] with D == 1 or 2
    Returns: models dict, discretizers dict, n_bins used, features_dim
    """
    # collect per-column arrays for fitting discretizers
    # we'll assume all sequences have same D
    sample_pat = next(iter(pattern_seqs))
    D = pattern_seqs[sample_pat][0].shape[1]
    if D > 2:
        raise ValueError("Discrete path supports at most 2 feature columns. Use continuous path for D>2.")

    # Build column-wise data
    all_col_data = [np.concatenate([s[:, i] for pat in pattern_seqs for s in pattern_seqs[pat]]) for i in range(D)]
    discretizers = {}
    binned_columns = []
    for i in range(D):
        disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        col_vals = all_col_data[i].reshape(-1, 1)
        disc.fit(col_vals)
        discretizers[i] = disc
        binned_columns.append(disc)

    # Helper to convert sequence (T,D) -> combined code (T,1)
    def seq_to_codes(seq):
        # seq shape (T, D)
        bins = []
        for i in range(D):
            b = discretizers[i].transform(seq[:, i].reshape(-1, 1)).astype(int).ravel()
            bins.append(b)
        if D == 1:
            codes = bins[0].reshape(-1, 1)
        else:
            codes = (bins[0] * n_bins + bins[1]).reshape(-1, 1)
        return codes

    # Prepare sequences per pattern into list of arrays for training
    seqs_per_pattern_codes = {}
    for pat, seqs in pattern_seqs.items():
        codes_list = [seq_to_codes(s) for s in seqs if len(s) > 0]
        seqs_per_pattern_codes[pat] = codes_list

    # Train a MultinomialHMM per pattern (with restarts)
    models = {}
    for pat, coded_seqs in seqs_per_pattern_codes.items():
        if not coded_seqs:
            continue
        X_concat = np.concatenate(coded_seqs)
        lengths = [len(s) for s in coded_seqs]
        best_model = None
        best_score = -np.inf
        for r in range(n_restarts):
            m = hmm.MultinomialHMM(n_components=n_states, n_iter=200, tol=1e-4, verbose=False, random_state=42 + r)
            # random initialization can help
            try:
                m.fit(X_concat, lengths)
            except Exception as e:
                # fallback: try uniform init then fit
                try:
                    m.startprob_ = np.ones(n_states) / n_states
                    m.transmat_ = np.ones((n_states, n_states)) / n_states
                    m.fit(X_concat, lengths)
                except Exception:
                    continue
            # score on training concat
            try:
                sc = np.sum([m.score(s) for s in coded_seqs])
            except Exception:
                sc = -np.inf
            if sc > best_score:
                best_score = sc
                best_model = m
        if best_model is None:
            continue
        # Smooth emission probs slightly to avoid zeros
        eps = 1e-8
        best_model.emissionprob_ = best_model.emissionprob_ + eps
        best_model.emissionprob_ = best_model.emissionprob_ / best_model.emissionprob_.sum(axis=1, keepdims=True)
        models[pat] = best_model

    return models, discretizers, n_bins, D

def train_continuous_hmms(pattern_seqs, n_states=8, pca_dim=6, n_restarts=1):
    """
    pattern_seqs: dict pattern -> [np.array(seq_timesteps x D)] with D >= 1
    We'll concatenate all data, fit StandardScaler + PCA, then train GaussianHMM per pattern on reduced data.
    """
    # gather all sequences to fit scaler + PCA
    all_seqs = [s for pat in pattern_seqs for s in pattern_seqs[pat]]
    if not all_seqs:
        raise ValueError("No sequences provided.")
    concat = np.vstack(all_seqs)
    scaler = StandardScaler().fit(concat)
    concat_scaled = scaler.transform(concat)

    # PCA
    pca_dim = min(pca_dim, concat_scaled.shape[1])
    pca = PCA(n_components=pca_dim).fit(concat_scaled)

    # prepare reduced sequences per pattern
    reduced_per_pattern = {}
    for pat, seqs in pattern_seqs.items():
        reduced = [pca.transform(scaler.transform(s)) for s in seqs if len(s) > 0]
        reduced_per_pattern[pat] = reduced

    # train GaussianHMM per pattern
    models = {}
    for pat, red_seqs in reduced_per_pattern.items():
        if not red_seqs:
            continue
        X_concat = np.vstack(red_seqs)
        lengths = [len(s) for s in red_seqs]
        best_model = None
        best_score = -np.inf
        for r in range(n_restarts):
            m = hmm.GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200, random_state=42+r)
            try:
                m.fit(X_concat, lengths)
            except Exception:
                continue
            try:
                sc = np.sum([m.score(s) for s in red_seqs])
            except Exception:
                sc = -np.inf
            if sc > best_score:
                best_score = sc
                best_model = m
        if best_model is not None:
            models[pat] = best_model

    # return models + scaler + pca for later transform
    return models, scaler, pca

# -----------------------
# Evaluation
# -----------------------
def evaluate_models_discrete(test_patterns, models, discretizers, n_bins, D):
    """
    test_patterns: dict pat -> [np.array(seq_timesteps x D)]
    models: dict pat -> MultinomialHMM trained on combined codes
    discretizers: dict index -> KBinsDiscretizer
    Returns metrics dict
    """
    total_seq = 0
    correct = 0
    confusion = Counter()
    total_next = 0
    next_correct = 0

    # helper: convert seq -> codes
    def seq_to_codes_local(seq):
        bins = []
        for i in range(D):
            bins.append(discretizers[i].transform(seq[:, i].reshape(-1, 1)).astype(int).ravel())
        if D == 1:
            codes = bins[0].reshape(-1, 1)
        else:
            codes = (bins[0] * n_bins + bins[1]).reshape(-1, 1)
        return codes

    # iterate test sequences
    for true_pat, seqs in test_patterns.items():
        for s in seqs:
            if len(s) < 2:
                continue
            codes = seq_to_codes_local(s)
            # classify by log-likelihood across models
            scores = {}
            for pat, m in models.items():
                try:
                    scores[pat] = m.score(codes)
                except Exception:
                    scores[pat] = -np.inf
            predicted = max(scores, key=scores.get)
            total_seq += 1
            if predicted == true_pat:
                correct += 1
            confusion[(true_pat, predicted)] += 1

            # next-delta prediction using predicted model
            m = models[predicted]
            for i in range(len(codes)-1):
                window = codes[:i+1]
                try:
                    _, states = m.decode(window, algorithm='viterbi')
                except Exception:
                    continue
                last_state = states[-1]
                next_obs_prob = m.transmat_[last_state].dot(m.emissionprob_)
                pred_code = int(np.argmax(next_obs_prob))
                if pred_code == int(codes[i+1][0]):
                    next_correct += 1
                total_next += 1

    classification_acc = correct / total_seq if total_seq else 0.0
    next_acc = next_correct / total_next if total_next else 0.0
    return {
        'classification_accuracy': classification_acc,
        'next_delta_accuracy': next_acc,
        'confusion': confusion,
        'total_sequences': total_seq,
    }

def evaluate_models_continuous(test_patterns, models, scaler, pca):
    total_seq = 0
    correct = 0
    confusion = Counter()
    # For continuous HMMs we'll only do classification (no discrete next-step)
    for true_pat, seqs in test_patterns.items():
        for s in seqs:
            if len(s) < 2:
                continue
            X = scaler.transform(s)
            Xr = pca.transform(X)
            scores = {}
            for pat, m in models.items():
                try:
                    scores[pat] = m.score(Xr)
                except Exception:
                    scores[pat] = -np.inf
            predicted = max(scores, key=scores.get)
            total_seq += 1
            if predicted == true_pat:
                correct += 1
            confusion[(true_pat, predicted)] += 1
    classification_acc = correct / total_seq if total_seq else 0.0
    return {
        'classification_accuracy': classification_acc,
        'confusion': confusion,
        'total_sequences': total_seq,
    }

# -----------------------
# Top-level orchestration
# -----------------------
def train_and_evaluate(train_csv, test_csv,
                       feature_cols=None,
                       label_col='pattern_type',
                       window_size=2000,
                       window_step=1000,
                       n_bins=16,
                       n_states=12,
                       do_log_scale=True):
    # Prepare training and test sequences
    train_patterns, train_df, feature_cols_used = prepare_sequences_from_csv(
        train_csv,
        feature_cols=feature_cols,
        label_col=label_col,
        window_size=window_size,
        window_step=window_step,
        do_log_scale=do_log_scale
    )
    test_patterns, test_df, _ = prepare_sequences_from_csv(
        test_csv,
        feature_cols=feature_cols,
        label_col=label_col,
        window_size=window_size,
        window_step=window_step,
        do_log_scale=do_log_scale
    )

    D = len(feature_cols_used)
    print(f"Using features: {feature_cols_used} (D={D})")
    print("Train sequences per pattern:")
    for pat in train_patterns:
        print(f"  {pat}: {len(train_patterns[pat])} sequences (example lengths: {[len(s) for s in train_patterns[pat]][:5]})")

    # Decide discrete vs continuous strategy
    if D <= 2:
        print("Using discrete Multinomial HMMs (per-column quantile binning + combined codes).")
        # adjust n_bins to keep combined code space manageable
        # if D==2, combined_symbols = n_bins * n_bins
        if D == 2 and n_bins > 32:
            print("Reducing n_bins to 32 to prevent blow-up")
            n_bins = 32
        models, discretizers, used_n_bins, used_D = train_discrete_hmms(
            train_patterns, n_bins=n_bins, n_states=n_states, n_restarts=3
        )
        print("Evaluating discrete models on test set...")
        metrics = evaluate_models_discrete(test_patterns, models, discretizers, used_n_bins, used_D)
        # Diagnostics printout
        print("Discrete HMM metrics:")
        print(f" Classification accuracy: {metrics['classification_accuracy']:.4f}")
        print(f" Next-delta accuracy: {metrics['next_delta_accuracy']:.4f}")
        print(" Confusion (true,pred) counts (top 20):")
        for k, v in metrics['confusion'].most_common(20):
            print(f"  {k}: {v}")
        return {
            'strategy': 'discrete',
            'models': models,
            'discretizers': discretizers,
            'metrics': metrics
        }
    else:
        print("Using continuous Gaussian HMMs (StandardScaler + PCA reduction).")
        models, scaler, pca = train_continuous_hmms(train_patterns, n_states=n_states, pca_dim=min(6, D), n_restarts=3)
        print("Evaluating continuous models on test set...")
        metrics = evaluate_models_continuous(test_patterns, models, scaler, pca)
        print("Continuous HMM metrics:")
        print(f" Classification accuracy: {metrics['classification_accuracy']:.4f}")
        print(" Confusion (true,pred) counts (top 20):")
        for k, v in metrics['confusion'].most_common(20):
            print(f"  {k}: {v}")
        return {
            'strategy': 'continuous',
            'models': models,
            'scaler': scaler,
            'pca': pca,
            'metrics': metrics
        }

# -----------------------
# CLI
# -----------------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--test_csv', required=True)
    p.add_argument('--features', nargs='*', default=None, help="feature column names (default: auto-detect Delta_with*)")
    p.add_argument('--window', type=int, default=2000)
    p.add_argument('--step', type=int, default=1000)
    p.add_argument('--n_bins', type=int, default=8)
    p.add_argument('--n_states', type=int, default=8)
    p.add_argument('--nolog', dest='do_log', action='store_false', help="disable log scaling")
    args = p.parse_args()
    result = train_and_evaluate(
        args.train_csv, args.test_csv,
        feature_cols=args.features,
        window_size=args.window, window_step=args.step,
        n_bins=args.n_bins, n_states=args.n_states,
        do_log_scale=args.do_log
    )
    print("\nDone. Summary metrics:")
    print(result['metrics'])

if __name__ == '__main__':
    cli()

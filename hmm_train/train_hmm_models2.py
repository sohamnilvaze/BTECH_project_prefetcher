import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict, Counter

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def train_HMM_models(df_path,n_bins,n_states):
    df = pd.read_csv(df_path)
    features_col = ["Delta_with_1_last_read", "Delta_with_1_last_write"]
    df[features_col] = np.sign(df[features_col]) * np.log1p(np.abs(df[features_col]))
    df[features_col] = df[features_col].fillna(0)

    # Clip extreme values (1st–99th percentile) jointly
    vals = df[features_col].values
    lo, hi = np.percentile(vals, [1, 99])
    vals_clipped = np.clip(vals, lo, hi)
    df[features_col] = vals_clipped

    # Quantize each feature column separately
    n_bins = n_bins
    discretizers = {}
    for col in features_col:
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        df[col] = disc.fit_transform(df[[col]]).astype(int)
        discretizers[col] = disc

    # Combine multi-feature bins → single integer symbol
    df["combined_code"] = df[features_col[0]] * n_bins + df[features_col[1]]

    # Group sequences by pattern_type
    sequences = defaultdict(list)
    for pattern, group in df.groupby("pattern_type"):
        seq = group["combined_code"].values.reshape(-1, 1)
        sequences[pattern].append(seq)

    # Train one HMM per pattern
    models = {}
    n_states = n_states
    for pattern, seq_list in sequences.items():
        X_concat = np.concatenate(seq_list)
        lengths = [len(s) for s in seq_list]

        model = hmm.MultinomialHMM(n_components=n_states, n_iter=150, random_state=42)
        model.fit(X_concat, lengths)
        epsilon = 1e-6
        model.emissionprob_ = model.emissionprob_ + epsilon
        model.emissionprob_ = model.emissionprob_ / model.emissionprob_.sum(axis=1, keepdims=True)
        models[pattern] = model
        print(f"Trained model for pattern {pattern}")

    return models, discretizers, n_bins, features_col


def classify_trace(trace_matrix, discretizers, models, n_bins, features_col):
    trace = np.array(trace_matrix, dtype=float)
    if trace.ndim == 1:
        trace = trace.reshape(-1, len(features_col))

    # Discretize each column
    binned_features = []
    for i, col in enumerate(features_col):
        disc = discretizers[col]
        binned = disc.transform(trace[:, i].reshape(-1, 1)).astype(int)
        binned_features.append(binned.flatten())

    # Combine into single code per timestep
    trace_combined = (binned_features[0] * n_bins + binned_features[1]).reshape(-1, 1)

    # Evaluate log-likelihood under each HMM
    scores = {}
    for pattern, model in models.items():
        try:
            log_likelihood = model.score(trace_combined)
        except:
            log_likelihood = -np.inf
        scores[pattern] = log_likelihood

    best_pattern = max(scores, key=scores.get)
    return best_pattern, scores, trace_combined


def predict_next_delta(trace_combined, model, discretizers, n_bins, features_col):
    logprob, states = model.decode(trace_combined, algorithm="viterbi")
    last_state = states[-1]

    # Predict next observation probabilities
    next_obs_prob = model.transmat_[last_state].dot(model.emissionprob_)

    predicted_code = np.argmax(next_obs_prob)
    predicted_bin_1 = predicted_code // n_bins
    predicted_bin_2 = predicted_code % n_bins

    # Decode bin edges to approximate delta values
    delta1 = discretizers[features_col[0]].bin_edges_[0][predicted_bin_1]
    delta2 = discretizers[features_col[1]].bin_edges_[0][predicted_bin_2]

    return (delta1, delta2), (predicted_bin_1, predicted_bin_2), next_obs_prob


def evaluate_hmm_models(test_csv, models, discretizers, n_bins, features_col):
    df = pd.read_csv(test_csv)
    df[features_col] = df[features_col].fillna(0)

    # Clip & discretize each column using trained discretizers
    binned_features = []
    for col in features_col:
        disc = discretizers[col]
        binned = disc.transform(df[[col]]).astype(int)
        binned_features.append(binned.flatten())

    trace_combined = (binned_features[0] * n_bins + binned_features[1]).reshape(-1, 1)
    pattern_types = df['pattern_type'].values

    total_sequences = 0
    correct_classification = 0
    next_delta_correct = 0
    total_deltas = 0
    confusion = Counter()

    # Group by pattern
    for pattern, group in df.groupby('pattern_type'):
        seq = (group[features_col[0]].values * n_bins + group[features_col[1]].values).reshape(-1, 1)

        # Classify sequence
        scores = {}
        for pat, model in models.items():
            try:
                scores[pat] = model.score(seq)
            except:
                scores[pat] = -np.inf
        predicted_pat = max(scores, key=scores.get)

        total_sequences += 1
        if predicted_pat == pattern:
            correct_classification += 1
        confusion[(pattern, predicted_pat)] += 1

        # Next-delta prediction
        model = models[predicted_pat]
        for i in range(len(seq) - 1):
            window = seq[:i+1]
            _, states = model.decode(window, algorithm="viterbi")
            last_state = states[-1]
            next_obs_prob = model.transmat_[last_state].dot(model.emissionprob_)
            predicted_code = np.argmax(next_obs_prob)
            if predicted_code == seq[i+1][0]:
                next_delta_correct += 1
            total_deltas += 1

    classification_accuracy = correct_classification / total_sequences
    next_delta_accuracy = next_delta_correct / total_deltas

    return {
        "classification_accuracy": classification_accuracy,
        "next_delta_accuracy": next_delta_accuracy,
        "confusion_matrix": confusion
    }


def main():
    # input_csv = input("Enter input lower GHB CSV filename: ").strip()
    input_csv = "ghb_1/f1_strided.csv"
    test_csv = "tests/test_std.csv"
    accuracies = []
    for n_bins in range(8,13):
        for n_states in range(16,33):
            models, discretizers, n_bins, features_col = train_HMM_models(input_csv,n_bins,n_states)
            results = evaluate_hmm_models(test_csv, models, discretizers, n_bins, features_col)
            accuracies.append(results["classification_accuracy"])



    

    # test_csv = input("Enter test CSV filename: ").strip()
    

    print("\n--- Evaluation Results ---")
    print(accuracies)
    # print("Classification accuracy:", results["classification_accuracy"])
    # print("Next-delta prediction accuracy:", results["next_delta_accuracy"])
    # print("Confusion matrix:", results["confusion_matrix"])


if __name__ == '__main__':
    main()

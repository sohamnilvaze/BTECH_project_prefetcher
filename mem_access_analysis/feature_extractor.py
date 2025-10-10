'''
Goal: turn raw delta columns into a compact feature vector per trace or per sliding window.These vectors are to be used for visualizing
Features computed:
1)mean_delta_read/mean_delta_write
2)median_delta_read/median_delta_write
3)mode_delta_read/mode_delta_write
4)unique count(no. of unique deltas in window)
5)zero_fraction(fraction of zero deltas)
6)entropy of histogram of deltas
7)autocorr_peak_flag
8)dominant frequency via FFT(index of max power)
9)percentile(25%,50%,75%)

compute entropy: p = counts/sum(counts) entropy = -sum(p * log2(p))
entropy-> how unpredictable the deltas are, lower the entropy the more predictable/regular the access pattern is

Autocorrelation: tells you how well a signal correlates with a lagged version of itself — essentially, whether deltas repeat after a fixed interval.A strong peak at lag = k means there’s a repeating pattern every k steps.
Example: stride accesses or repeated loops.

FFT converts the sequence of deltas into the frequency domain, revealing how often certain stride patterns repeat.The dominant frequency tells you the rate of change or stride repetition frequency.

'''
import numpy as np
import pandas as pd
from scipy.stats import entropy, mode

def window_seq(arr, window, step):
    n = len(arr)
    if n == 0: return []
    starts = list(range(0, max(1, n - window + 1), step))
    if not starts: starts = [0]
    return [arr[s:s+window] for s in starts]

def compute_window_features(delta_read, delta_write=None):
    """
    delta_read: 1D numpy array of deltas for window
    returns dict of features
    """
    f = {}
    x = np.array(delta_read, dtype=float)
    if len(x)==0:
        return None
    # basic stats
    f['mean'] = x.mean()
    f['std'] = x.std()
    f['median'] = np.median(x)
    f['unique'] = len(np.unique(x))
    f['zeros_frac'] = np.mean(x==0)

    # histogram + entropy
    counts, edges = np.histogram(x, bins='auto')
    probs = counts / (counts.sum() + 1e-12)
    f['entropy'] = entropy(probs + 1e-12, base=2)

    # mode (most common delta)
    try:
        m = mode(x, keepdims=False)
        f['mode'] = float(m.mode)
        f['mode_count'] = int(m.count)
    except:
        f['mode'] = 0.0
        f['mode_count'] = 0

    # autocorrelation -> find first peak lag (excluding lag 0)
    x0 = x - x.mean()
    ac = np.correlate(x0, x0, mode='full')
    ac = ac[len(ac)//2:]  # non-negative lags
    if ac.sum() == 0:
        f['ac_first_peak'] = 0
    else:
        # normalize and find peaks
        acn = ac / (ac[0] + 1e-12)
        # find lag >0 where acn has local maxima > 0.2
        peaks = np.where((acn[1:-1] > acn[:-2]) & (acn[1:-1] > acn[2:]))[0] + 1
        strong = [p for p in peaks if acn[p] > 0.2]
        f['ac_first_peak'] = int(strong[0]) if strong else 0

    # FFT dominant freq (index)
    fftp = np.abs(np.fft.rfft(x - x.mean()))
    if len(fftp) > 1:
        dom = np.argmax(fftp[1:]) + 1
        f['fft_dom_bin'] = int(dom)
        f['fft_dom_power'] = float(fftp[dom])
    else:
        f['fft_dom_bin'] = 0
        f['fft_dom_power'] = 0.0

    # optionally handle writes similarly
    if delta_write is not None:
        y = np.array(delta_write, dtype=float)
        f['w_mean'] = y.mean(); f['w_std'] = y.std()

    return f

def extract_features_from_df(df, feature_read_cols, feature_write_cols=None,
                             window=500, step=250, do_log=False):
    rows = []
    # prepare a combined vector (choose primary column for temporal order)
    # assume df is already filtered to one program/pattern
    for start in range(0, max(1, len(df) - window + 1), step):
        win = df.iloc[start:start+window]
        # choose primary series: Delta_with_1_last_read over time
        # if multiple read columns, you could use last_1 only as representative
        dr = win[feature_read_cols[0]].values
        dw = None
        if feature_write_cols:
            dw = win[feature_write_cols[0]].values
        if do_log:
            # log-scale to compress large jumps
            dr = np.sign(dr) * np.log1p(np.abs(dr))
            if dw is not None:
                dw = np.sign(dw) * np.log1p(np.abs(dw))
        f = compute_window_features(dr, dw)
        if f is None: continue
        f['start'] = start
        f['length'] = len(dr)
        rows.append(f)
    return pd.DataFrame(rows)



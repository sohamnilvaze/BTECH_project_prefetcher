'''
Goal: produce informative plots for each trace or per-window features that help you visually distinguish access types.
Plots:
1)Histogram of deltas
2)Time series of deltas
3)ENtropy v/s time
4)Autocorrelation plot per trace
5)FFT power spectrum to find periodic patterns
6)ter or PCA/TSNE of per-window feature vectors across different programs to see clustering.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_delta_histogram(deltas, bins=100, title=None):
    plt.figure(figsize=(6,3))
    sns.histplot(deltas, bins=bins, kde=False)
    plt.title(title or "Delta histogram")
    plt.xlabel("Delta")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_timeseries(deltas, downsample=1, title=None):
    plt.figure(figsize=(8,2))
    plt.plot(deltas[::downsample], marker='.', linestyle='-', alpha=0.7)
    plt.title(title or "Delta timeseries")
    plt.xlabel("Access index")
    plt.ylabel("Delta")
    plt.tight_layout()
    plt.show()

def plot_entropy_timeline(windowed_features):
    plt.figure(figsize=(8,2))
    plt.plot(windowed_features['entropy'].values, marker='o')
    plt.title("Entropy over windows")
    plt.xlabel("window idx")
    plt.ylabel("entropy (bits)")
    plt.tight_layout()
    plt.show()

def plot_rolling_means(df,feature_col):
    rolling_mean = df[feature_col].rolling(100).mean()
    rolling_std = df[feature_col].rolling(100).mean()
    index = df.index
    plt.plot(index,rolling_mean,color = 'green')
    plt.plot(index,rolling_std,color = 'red')
    plt.title(f"Rolling means, std for {feature_col}")
    plt.show()

def plot_corr_heatmap(df,feature_cols):
    corr = df[feature_cols].corr()
    sns.heatmap(corr,annot=True, cmap="coolwarm")
    plt.title(f"Heatmap for {feature_cols}")
    plt.show()

def plot_windowed_entropy(feats):
    plt.scatter(feats['mean'],feats['entropy'],c = range(len(feats)),cmap="viridis")
    plt.ylabel("Entropy (bits)")
    plt.xlabel("Mean")
    plt.title("Entropy v/s mean")
    plt.show()

def plot_kde(df,col):
    sns.kdeplot(df[col],bw_adjust=0.5)
    plt.title(f"Kernel density plot for {col}")
    plt.show()

def plot_cdf(df,col):
    sorted_deltas = np.sort(df[col])
    cdf = np.arange(len(sorted_deltas)) / float(len(sorted_deltas))
    plt.plot(sorted_deltas,cdf)
    plt.title(f"CDF for {col}")
    plt.show()

def address_space_coverage(df,col):
    plt.plot(np.cumsum(np.abs(df[col])))
    plt.title(f"Cummulative sum for {col}")
    plt.show()

def fft_analysis(df,col):
    from scipy.fft import rfft, rfftfreq
    y = df[col].values
    yf = np.abs(rfft(y - np.mean(y)))
    xf = rfftfreq(len(y),d = 1)
    plt.plot(xf,yf)
    plt.show()

#Not able to
def autocorrelation_plot(df,col):
    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df[col])
    plt.title(f"Autocorrelation plot for {col}")
    plt.show()

def read_vs_write(df):
    plt.scatter(df['Delta_with_1_last_read'],df['Delta_with_1_last_write'],alpha = 0.5)
    plt.xlabel("read Deltas")
    plt.ylabel("write deltas")
    plt.title("Read v/s write deltas")
    plt.show()

def plot_all_plots(df,col):
    plot_kde(df,col)
    plot_cdf(df,col)
    address_space_coverage(df,col)
    #fft_analysis(df,col)
    autocorrelation_plot(df,col)
    read_vs_write(df)

def plot_inter_arrival(df, time_col='Timestamp'):
    times = df[time_col].sort_values().values
    inter_arrival = np.diff(times)
    plt.figure(figsize=(6,3))
    plt.hist(inter_arrival, bins=50, color='dodgerblue', alpha=0.7)
    plt.xlabel('Inter-arrival time (steps)')
    plt.ylabel('Count')
    plt.title('Histogram of Inter-arrival Times')
    plt.tight_layout()
    plt.show()

def plot_inter_arrival_timeseries(df, time_col='Timestamp'):
    times = df[time_col].sort_values().values
    inter_arrival = np.diff(times)
    plt.figure(figsize=(8,2))
    plt.plot(inter_arrival, marker='.', linestyle='-')
    plt.xlabel('Access Index')
    plt.ylabel('Inter-arrival Time')
    plt.title('Inter-arrival Timeseries')
    plt.tight_layout()
    plt.show()


def plot_pca_features(feats, label_col='type'):
    pca = PCA(n_components=2)
    X = feats.drop(columns=[label_col]).values
    y = feats[label_col].values
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='Set2')
    plt.title('PCA of Per-Window Features')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

def plot_tsne_features(feats, label_col='type'):
    X = feats.drop(columns=[label_col]).values
    y = feats[label_col].values
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette='Set2')
    plt.title('t-SNE of Per-Window Features')
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.show()

def plot_feature_pairplot(feats, label_col='type'):
    sns.pairplot(feats, hue=label_col, diag_kind='kde', plot_kws=dict(alpha=0.5))
    plt.suptitle('Feature Pairplot', y=1.02)
    plt.show()








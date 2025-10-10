from trace_loader import load_trace_csv, sanitize_df
from feature_extractor import extract_features_from_df
from pattern_visualizer import plot_delta_histogram, plot_timeseries, plot_entropy_timeline, plot_rolling_means, plot_corr_heatmap, plot_windowed_entropy, plot_all_plots

name = "ba_1"
dfs = load_trace_csv(f"../mem_access_traces/{name}.csv")
feature_cols = ["Delta_with_1_last_read","Delta_with_1_last_write"]


for name, df in dfs.items():
    df = sanitize_df(df,feature_cols=feature_cols)
    feats = extract_features_from_df(df,["Delta_with_1_last_read"],["Delta_with_1_last_write"],window = 500,step = 250,do_log=True)
    feats.to_csv(f"../mem_access_traces/features_windowed_{name}.csv",index = False)
    plot_delta_histogram(df['Delta_with_1_last_read'], title=f"{name}- 1 last read")
    plot_delta_histogram(df['Delta_with_1_last_write'], title=f"{name}- 1 last write")
    plot_timeseries(df['Delta_with_1_last_read'],title=f"{name}- 1 last read")
    plot_timeseries(df['Delta_with_1_last_write'],title=f"{name}- 1 last write")
    plot_entropy_timeline(feats)
    plot_rolling_means(df,'Delta_with_1_last_read')
    plot_rolling_means(df,'Delta_with_1_last_write')
    plot_corr_heatmap(df,['Delta_with_1_last_read','Delta_with_2_last_read','Delta_with_3_last_read'])
    plot_corr_heatmap(df,['Delta_with_1_last_write','Delta_with_2_last_write','Delta_with_3_last_write'])
    plot_all_plots(df,['Delta_with_1_last_read'])
    plot_all_plots(df,['Delta_with_1_last_write'])
    



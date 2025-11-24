import argparse
from trace_loader import load_trace_csv, sanitize_df
from feature_extractor import extract_features_from_df
from pattern_visualizer import *



parser = argparse.ArgumentParser(description="Extract features from trace CSVs.")
parser.add_argument('name', help="Base name of the trace file (e.g., std_128)")
args = parser.parse_args()

name = args.name
dfs = load_trace_csv(f"{name}")
feature_cols = ["Delta_with_1_last_read","Delta_with_1_last_write","Delta_with_2_last_read","Delta_with_2_last_write","Delta_with_3_last_read","Delta_with_3_last_write","Delta_with_4_last_read","Delta_with_4_last_write","Delta_with_5_last_read","Delta_with_5_last_write","Delta_with_6_last_read","Delta_with_6_last_write","Delta_with_7_last_read","Delta_with_7_last_write","Delta_with_8_last_read","Delta_with_8_last_write"]


for name, df in dfs.items():
    df, n_r = sanitize_df(df,feature_cols=feature_cols)
    win = 100
    step = 50
    feats = extract_features_from_df(df,["Delta_with_1_last_read","Delta_with_2_last_read","Delta_with_3_last_read","Delta_with_4_last_read"],["Delta_with_1_last_write","Delta_with_2_last_write","Delta_with_3_last_write","Delta_with_4_last_write"],window = win,step = step,do_log=True)
    feats.to_csv(f"../mem_access_traces/features_{name}.csv",index = False)
    # plot_delta_histogram(df['Delta_with_1_last_read'], title=f"{name}- 1 last read")
    # plot_delta_histogram(df['Delta_with_1_last_write'], title=f"{name}- 1 last write")
    # plot_timeseries(df['Delta_with_1_last_read'],title=f"{name}- 1 last read")
    # plot_timeseries(df['Delta_with_1_last_write'],title=f"{name}- 1 last write")
    # plot_entropy_timeline(feats)
    # plot_rolling_means(df,'Delta_with_1_last_read')
    # plot_rolling_means(df,'Delta_with_1_last_write')
    # plot_corr_heatmap(df,['Delta_with_1_last_read','Delta_with_2_last_read','Delta_with_3_last_read'])
    # plot_corr_heatmap(df,['Delta_with_1_last_write','Delta_with_2_last_write','Delta_with_3_last_write'])
    # plot_all_plots(df,['Delta_with_1_last_read'])
    # plot_all_plots(df,['Delta_with_1_last_write'])
    # df_R = df[df['Type'] == 'R']
    # df_W = df[df['Type'] == 'W']
    # plot_inter_arrival(df_R)
    # plot_inter_arrival_timeseries(df_R)
    # plot_inter_arrival(df_W)
    # plot_inter_arrival_timeseries(df_W)



    



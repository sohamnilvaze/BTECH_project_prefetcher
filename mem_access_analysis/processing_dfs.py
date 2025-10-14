import pandas as pd

#1- block access
#2- sequantial access
#3- strided access
#4- column major
#5- row major
#6- indirect access

def add_target_col():
    for dfs in ["../mem_access_traces/features_ba_1.csv","../mem_access_traces/features_ba_2.csv","../mem_access_traces/features_ba_4.csv","../mem_access_traces/features_ba_8.csv","../mem_access_traces/features_ba_16.csv","../mem_access_traces/features_ba_32.csv","../mem_access_traces/features_ba_64.csv","../mem_access_traces/features_ba_128.csv"] :
        df = pd.read_csv(dfs)
        df["Target"] = 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    for dfs in ["../mem_access_traces/features_saif_400.csv","../mem_access_traces/features_saib_400.csv","../mem_access_traces/features_sadf_400.csv","../mem_access_traces/features_sadb_400.csv","../mem_access_traces/features_saif_200.csv","../mem_access_traces/features_saib_200.csv","../mem_access_traces/features_sadf_200.csv","../mem_access_traces/features_sadb_200.csv","../mem_access_traces/features_saif_100.csv","../mem_access_traces/features_saib_100.csv","../mem_access_traces/features_sadf_100.csv","../mem_access_traces/features_sadb_100.csv"] :
        df = pd.read_csv(dfs)
        df["Target"] = 2
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    for dfs in ["../mem_access_traces/features_std_1.csv","../mem_access_traces/features_std_2.csv","../mem_access_traces/features_std_4.csv","../mem_access_traces/features_std_8.csv","../mem_access_traces/features_std_16.csv","../mem_access_traces/features_std_32.csv","../mem_access_traces/features_std_64.csv","../mem_access_traces/features_std_128.csv"] : 
        df = pd.read_csv(dfs)
        df["Target"] = 3
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    for dfs in ["../mem_access_traces/features_mcol_1_100.csv","../mem_access_traces/features_mcol_10_90.csv","../mem_access_traces/features_mcol_20_80.csv","../mem_access_traces/features_mcol_30_70.csv","../mem_access_traces/features_mcol_40_60.csv","../mem_access_traces/features_mcol_50_50.csv","../mem_access_traces/features_mcol_60_40.csv","../mem_access_traces/features_mcol_70_30.csv","../mem_access_traces/features_mcol_80_20.csv","../mem_access_traces/features_mcol_90_10.csv","../mem_access_traces/features_mcol_100_1.csv"] :
        df = pd.read_csv(dfs)
        df["Target"] = 4
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    for dfs in ["../mem_access_traces/features_mrow_1_100.csv","../mem_access_traces/features_mrow_10_90.csv","../mem_access_traces/features_mrow_20_80.csv","../mem_access_traces/features_mrow_30_70.csv","../mem_access_traces/features_mrow_40_60.csv","../mem_access_traces/features_mrow_50_50.csv","../mem_access_traces/features_mrow_60_40.csv","../mem_access_traces/features_mrow_70_30.csv","../mem_access_traces/features_mrow_80_20.csv","../mem_access_traces/features_mrow_90_10.csv","../mem_access_traces/features_mrow_100_1.csv"] :
        df = pd.read_csv(dfs)
        df["Target"] = 5
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    for dfs in ["../mem_access_traces/features_ia_3.csv","../mem_access_traces/features_ia_10.csv","../mem_access_traces/features_ia_50.csv","../mem_access_traces/features_ia_100.csv","../mem_access_traces/features_ia_250.csv","../mem_access_traces/features_ia_500.csv","../mem_access_traces/features_ia_750.csv","../mem_access_traces/features_ia_1000.csv"]:
        df = pd.read_csv(dfs)
        df["Target"] = 6
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")

        

def merge_dfs():
    #1
    df1 = pd.read_csv("../mem_access_traces/features_ba_1.csv")
    df2 = pd.read_csv("../mem_access_traces/features_ba_2.csv")
    df3 = pd.read_csv("../mem_access_traces/features_ba_4.csv")
    df4 = pd.read_csv("../mem_access_traces/features_ba_8.csv")
    df5 = pd.read_csv("../mem_access_traces/features_ba_16.csv")
    df6 = pd.read_csv("../mem_access_traces/features_ba_32.csv")
    df7 = pd.read_csv("../mem_access_traces/features_ba_64.csv")
    df8 = pd.read_csv("../mem_access_traces/features_ba_128.csv")

    #2
    df9 = pd.read_csv("../mem_access_traces/features_saif_400.csv")
    df10 = pd.read_csv("../mem_access_traces/features_saib_400.csv")
    df11 = pd.read_csv("../mem_access_traces/features_sadf_400.csv")
    df12 = pd.read_csv("../mem_access_traces/features_sadb_400.csv")
    df13 = pd.read_csv("../mem_access_traces/features_saif_200.csv")
    df14 = pd.read_csv("../mem_access_traces/features_saib_200.csv")
    df15 = pd.read_csv("../mem_access_traces/features_sadf_200.csv")
    df16 = pd.read_csv("../mem_access_traces/features_sadb_200.csv")
    df17 = pd.read_csv("../mem_access_traces/features_saif_100.csv")
    df18 = pd.read_csv("../mem_access_traces/features_saib_100.csv")
    df19 = pd.read_csv("../mem_access_traces/features_sadf_100.csv")
    df20 = pd.read_csv("../mem_access_traces/features_sadb_100.csv")

    #3
    df21 = pd.read_csv("../mem_access_traces/features_std_1.csv")
    df22 = pd.read_csv("../mem_access_traces/features_std_2.csv")
    df23 = pd.read_csv("../mem_access_traces/features_std_4.csv")
    df24 = pd.read_csv("../mem_access_traces/features_std_8.csv")
    df25 = pd.read_csv("../mem_access_traces/features_std_16.csv")
    df26 = pd.read_csv("../mem_access_traces/features_std_32.csv")
    df27 = pd.read_csv("../mem_access_traces/features_std_64.csv")
    df28 = pd.read_csv("../mem_access_traces/features_std_128.csv")

    #6
    df29 = pd.read_csv("../mem_access_traces/features_ia_3.csv")
    df30 = pd.read_csv("../mem_access_traces/features_ia_10.csv")
    df31 = pd.read_csv("../mem_access_traces/features_ia_50.csv")
    df32 = pd.read_csv("../mem_access_traces/features_ia_100.csv")
    df33 = pd.read_csv("../mem_access_traces/features_ia_250.csv")
    df34 = pd.read_csv("../mem_access_traces/features_ia_500.csv")
    df35 = pd.read_csv("../mem_access_traces/features_ia_750.csv")
    df36 = pd.read_csv("../mem_access_traces/features_ia_1000.csv")

    #4
    df37 = pd.read_csv("../mem_access_traces/features_mcol_1_100.csv")
    df38 = pd.read_csv("../mem_access_traces/features_mcol_10_90.csv")
    df39 = pd.read_csv("../mem_access_traces/features_mcol_20_80.csv")
    df40 = pd.read_csv("../mem_access_traces/features_mcol_30_70.csv")
    df41 = pd.read_csv("../mem_access_traces/features_mcol_40_60.csv")
    df42 = pd.read_csv("../mem_access_traces/features_mcol_50_50.csv")
    df43 = pd.read_csv("../mem_access_traces/features_mcol_60_40.csv")
    df44 = pd.read_csv("../mem_access_traces/features_mcol_70_30.csv")
    df45 = pd.read_csv("../mem_access_traces/features_mcol_80_20.csv")
    df46 = pd.read_csv("../mem_access_traces/features_mcol_90_10.csv")
    df47 = pd.read_csv("../mem_access_traces/features_mcol_100_1.csv")

    #5
    df48 = pd.read_csv("../mem_access_traces/features_mrow_1_100.csv")
    df49 = pd.read_csv("../mem_access_traces/features_mrow_10_90.csv")
    df50 = pd.read_csv("../mem_access_traces/features_mrow_20_80.csv")
    df51 = pd.read_csv("../mem_access_traces/features_mrow_30_70.csv")
    df52 = pd.read_csv("../mem_access_traces/features_mrow_40_60.csv")
    df53 = pd.read_csv("../mem_access_traces/features_mrow_50_50.csv")
    df54 = pd.read_csv("../mem_access_traces/features_mrow_60_40.csv")
    df55 = pd.read_csv("../mem_access_traces/features_mrow_70_30.csv")
    df56 = pd.read_csv("../mem_access_traces/features_mrow_80_20.csv")
    df57 = pd.read_csv("../mem_access_traces/features_mrow_90_10.csv")
    df58 = pd.read_csv("../mem_access_traces/features_mrow_100_1.csv")







    all_dfs = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df26,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58]

    merged_df = pd.concat(all_dfs, axis=0,ignore_index=True)
    print(merged_df.shape)
    
    merged_df.to_csv("../mem_access_traces/merged.csv",index=False)


add_target_col()
merge_dfs()
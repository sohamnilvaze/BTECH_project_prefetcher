import pandas as pd
import os

#1- block access
#2- sequantial access
#3- strided access
#4- column major
#5- row major
#6- indirect access
#7- reuse
#8- reverse linked list
#9- random access
#10- recursive
#11- linked list

def add_target_col():
    i = 1
    for dfs in ["../mem_access_traces/features_ba_1.csv","../mem_access_traces/features_ba_2.csv","../mem_access_traces/features_ba_4.csv","../mem_access_traces/features_ba_8.csv","../mem_access_traces/features_ba_16.csv","../mem_access_traces/features_ba_32.csv","../mem_access_traces/features_ba_64.csv","../mem_access_traces/features_ba_128.csv"] : #8
        df = pd.read_csv(dfs)
        df["Target"] = 1
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_saif_400.csv","../mem_access_traces/features_saib_400.csv","../mem_access_traces/features_sadf_400.csv","../mem_access_traces/features_sadb_400.csv","../mem_access_traces/features_saif_200.csv","../mem_access_traces/features_saib_200.csv","../mem_access_traces/features_sadf_200.csv","../mem_access_traces/features_sadb_200.csv","../mem_access_traces/features_saif_100.csv","../mem_access_traces/features_saib_100.csv","../mem_access_traces/features_sadf_100.csv","../mem_access_traces/features_sadb_100.csv"] : #12
        df = pd.read_csv(dfs)
        df["Target"] = 2
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_std_1.csv","../mem_access_traces/features_std_2.csv","../mem_access_traces/features_std_4.csv","../mem_access_traces/features_std_8.csv","../mem_access_traces/features_std_16.csv","../mem_access_traces/features_std_32.csv","../mem_access_traces/features_std_64.csv","../mem_access_traces/features_std_128.csv"] : #8
        df = pd.read_csv(dfs)
        df["Target"] = 3
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_mcol_1_100.csv","../mem_access_traces/features_mcol_10_90.csv","../mem_access_traces/features_mcol_20_80.csv","../mem_access_traces/features_mcol_30_70.csv","../mem_access_traces/features_mcol_40_60.csv","../mem_access_traces/features_mcol_50_50.csv","../mem_access_traces/features_mcol_60_40.csv","../mem_access_traces/features_mcol_70_30.csv","../mem_access_traces/features_mcol_80_20.csv","../mem_access_traces/features_mcol_90_10.csv","../mem_access_traces/features_mcol_100_1.csv"] : #11
        df = pd.read_csv(dfs)
        df["Target"] = 4
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_mrow_1_100.csv","../mem_access_traces/features_mrow_10_90.csv","../mem_access_traces/features_mrow_20_80.csv","../mem_access_traces/features_mrow_30_70.csv","../mem_access_traces/features_mrow_40_60.csv","../mem_access_traces/features_mrow_50_50.csv","../mem_access_traces/features_mrow_60_40.csv","../mem_access_traces/features_mrow_70_30.csv","../mem_access_traces/features_mrow_80_20.csv","../mem_access_traces/features_mrow_90_10.csv","../mem_access_traces/features_mrow_100_1.csv"] : #11
        df = pd.read_csv(dfs)
        df["Target"] = 5
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_ia_3.csv","../mem_access_traces/features_ia_10.csv","../mem_access_traces/features_ia_50.csv","../mem_access_traces/features_ia_100.csv","../mem_access_traces/features_ia_250.csv","../mem_access_traces/features_ia_500.csv","../mem_access_traces/features_ia_750.csv","../mem_access_traces/features_ia_1000.csv"]: #8
        df = pd.read_csv(dfs)
        df["Target"] = 6
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in["../mem_access_traces/features_reu_10.csv","../mem_access_traces/features_reu_20.csv","../mem_access_traces/features_reu_30.csv","../mem_access_traces/features_reu_40.csv","../mem_access_traces/features_reu_50.csv","../mem_access_traces/features_reu_60.csv","../mem_access_traces/features_reu_70.csv","../mem_access_traces/features_reu_80.csv","../mem_access_traces/features_reu_90.csv"]: #9
        df = pd.read_csv(dfs)
        df["Target"] = 7
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_rll_2.csv","../mem_access_traces/features_rll_4.csv","../mem_access_traces/features_rll_8.csv","../mem_access_traces/features_rll_16.csv","../mem_access_traces/features_rll_32.csv","../mem_access_traces/features_rll_64.csv","../mem_access_traces/features_rll_128.csv","../mem_access_traces/features_rll_256.csv","../mem_access_traces/features_rll_512.csv","../mem_access_traces/features_rll_1024.csv"]: #10
        df = pd.read_csv(dfs)
        df["Target"] = 8
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_ra_10.csv","../mem_access_traces/features_ra_50.csv","../mem_access_traces/features_ra_100.csv","../mem_access_traces/features_ra_250.csv","../mem_access_traces/features_ra_500.csv","../mem_access_traces/features_ra_750.csv","../mem_access_traces/features_ra_1000.csv"]: #7
        df = pd.read_csv(dfs)
        df["Target"] = 9
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_rec_10.csv","../mem_access_traces/features_rec_50.csv","../mem_access_traces/features_rec_100.csv","../mem_access_traces/features_rec_250.csv","../mem_access_traces/features_rec_500.csv","../mem_access_traces/features_rec_750.csv","../mem_access_traces/features_rec_1000.csv"]: #7
        df = pd.read_csv(dfs)
        df["Target"] = 10
        df["Fine_grained_Target"] = i
        i = i + 1
        df.to_csv(dfs,index=False)
        print(f"Added for {dfs}")
    i = 1
    for dfs in ["../mem_access_traces/features_ll_2.csv","../mem_access_traces/features_ll_4.csv","../mem_access_traces/features_ll_8.csv","../mem_access_traces/features_ll_16.csv","../mem_access_traces/features_ll_32.csv","../mem_access_traces/features_ll_64.csv","../mem_access_traces/features_ll_128.csv","../mem_access_traces/features_ll_256.csv","../mem_access_traces/features_ll_512.csv","../mem_access_traces/features_ll_1024.csv"]: #10
        df = pd.read_csv(dfs)
        df["Target"] = 11
        df["Fine_grained_Target"] = i
        i = i + 1
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

    ba_dfs = [df1,df2,df3,df4,df5,df6,df7,df8]
    ba_merged_df = pd.concat(ba_dfs, axis=0,ignore_index=True)
    ba_merged_df.to_csv("../mem_access_traces/ba_merged4.csv",index=False)


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

    sa_dfs = [df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20]
    sa_merged_df = pd.concat(sa_dfs, axis=0,ignore_index=True)
    sa_merged_df.to_csv("../mem_access_traces/sa_merged4.csv",index=False)

    #3
    df21 = pd.read_csv("../mem_access_traces/features_std_1.csv")
    df22 = pd.read_csv("../mem_access_traces/features_std_2.csv")
    df23 = pd.read_csv("../mem_access_traces/features_std_4.csv")
    df24 = pd.read_csv("../mem_access_traces/features_std_8.csv")
    df25 = pd.read_csv("../mem_access_traces/features_std_16.csv")
    df26 = pd.read_csv("../mem_access_traces/features_std_32.csv")
    df27 = pd.read_csv("../mem_access_traces/features_std_64.csv")
    df28 = pd.read_csv("../mem_access_traces/features_std_128.csv")

    std_dfs = [df21,df22,df23,df24,df25,df26,df27,df28]
    std_merged_df = pd.concat(std_dfs, axis=0,ignore_index=True)
    std_merged_df.to_csv("../mem_access_traces/std_merged4.csv",index=False)

    #6
    df29 = pd.read_csv("../mem_access_traces/features_ia_3.csv")
    df30 = pd.read_csv("../mem_access_traces/features_ia_10.csv")
    df31 = pd.read_csv("../mem_access_traces/features_ia_50.csv")
    df32 = pd.read_csv("../mem_access_traces/features_ia_100.csv")
    df33 = pd.read_csv("../mem_access_traces/features_ia_250.csv")
    df34 = pd.read_csv("../mem_access_traces/features_ia_500.csv")
    df35 = pd.read_csv("../mem_access_traces/features_ia_750.csv")
    df36 = pd.read_csv("../mem_access_traces/features_ia_1000.csv")

    ia_dfs = [df29,df30,df31,df32,df33,df34,df35,df36]
    ia_merged_df = pd.concat(ia_dfs, axis=0,ignore_index=True)
    ia_merged_df.to_csv("../mem_access_traces/ia_merged4.csv",index=False)

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

    mcol_dfs = [df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47]
    mcol_merged_df = pd.concat(mcol_dfs, axis=0,ignore_index=True)
    mcol_merged_df.to_csv("../mem_access_traces/mcol_merged4.csv",index=False)

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

    mrow_dfs = [df48,df49,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58]
    mrow_merged_df = pd.concat(mrow_dfs, axis=0,ignore_index=True)
    mrow_merged_df.to_csv("../mem_access_traces/mrow_merged4.csv",index=False)

    #7
    df59 = pd.read_csv("../mem_access_traces/features_reu_10.csv")
    df60 = pd.read_csv("../mem_access_traces/features_reu_20.csv")
    df61 = pd.read_csv("../mem_access_traces/features_reu_30.csv")
    df62 = pd.read_csv("../mem_access_traces/features_reu_40.csv")
    df63 = pd.read_csv("../mem_access_traces/features_reu_50.csv")
    df64 = pd.read_csv("../mem_access_traces/features_reu_60.csv")
    df65 = pd.read_csv("../mem_access_traces/features_reu_70.csv")
    df66 = pd.read_csv("../mem_access_traces/features_reu_80.csv")
    df67 = pd.read_csv("../mem_access_traces/features_reu_90.csv")

    reu_dfs = [df59,df60,df61,df62,df63,df64,df65,df66,df67]
    reu_merged_df = pd.concat(reu_dfs, axis=0,ignore_index=True)
    reu_merged_df.to_csv("../mem_access_traces/reu_merged4.csv",index=False)

    #8
    df68 = pd.read_csv("../mem_access_traces/features_rll_2.csv")
    df69 = pd.read_csv("../mem_access_traces/features_rll_4.csv")
    df70 = pd.read_csv("../mem_access_traces/features_rll_8.csv")
    df71 = pd.read_csv("../mem_access_traces/features_rll_16.csv")
    df72 = pd.read_csv("../mem_access_traces/features_rll_32.csv")
    df73 = pd.read_csv("../mem_access_traces/features_rll_64.csv")
    df74 = pd.read_csv("../mem_access_traces/features_rll_128.csv")
    df75 = pd.read_csv("../mem_access_traces/features_rll_256.csv")
    df76 = pd.read_csv("../mem_access_traces/features_rll_512.csv")
    df77 = pd.read_csv("../mem_access_traces/features_rll_1024.csv")

    rll_dfs = [df68,df69,df70,df71,df72,df73,df74,df75,df76,df77]
    rll_merged_df = pd.concat(rll_dfs, axis=0,ignore_index=True)
    rll_merged_df.to_csv("../mem_access_traces/rll_merged4.csv",index=False)

    #9
    df78 = pd.read_csv("../mem_access_traces/features_ra_10.csv")
    df79 = pd.read_csv("../mem_access_traces/features_ra_50.csv")
    df80 = pd.read_csv("../mem_access_traces/features_ra_100.csv")
    df81 = pd.read_csv("../mem_access_traces/features_ra_250.csv")
    df82 = pd.read_csv("../mem_access_traces/features_ra_500.csv")
    df83 = pd.read_csv("../mem_access_traces/features_ra_750.csv")
    df84 = pd.read_csv("../mem_access_traces/features_ra_1000.csv")

    ra_dfs = [df78,df79,df80,df81,df82,df83,df84]
    ra_merged_df = pd.concat(ra_dfs, axis=0,ignore_index=True)
    ra_merged_df.to_csv("../mem_access_traces/ra_merged4.csv",index=False)

    #10
    df85 = pd.read_csv("../mem_access_traces/features_rec_10.csv")
    df86 = pd.read_csv("../mem_access_traces/features_rec_50.csv")
    df87 = pd.read_csv("../mem_access_traces/features_rec_100.csv")
    df88 = pd.read_csv("../mem_access_traces/features_rec_250.csv")
    df89 = pd.read_csv("../mem_access_traces/features_rec_500.csv")
    df90 = pd.read_csv("../mem_access_traces/features_rec_750.csv")
    df91 = pd.read_csv("../mem_access_traces/features_rec_1000.csv")

    rec_dfs = [df85,df86,df87,df88,df89,df6,df90,df91]
    rec_merged_df = pd.concat(rec_dfs, axis=0,ignore_index=True)
    rec_merged_df.to_csv("../mem_access_traces/rec_merged4.csv",index=False)

    #11
    df92 = pd.read_csv("../mem_access_traces/features_ll_2.csv")
    df93 = pd.read_csv("../mem_access_traces/features_ll_4.csv")
    df94 = pd.read_csv("../mem_access_traces/features_ll_8.csv")
    df95 = pd.read_csv("../mem_access_traces/features_ll_16.csv")
    df96 = pd.read_csv("../mem_access_traces/features_ll_32.csv")
    df97 = pd.read_csv("../mem_access_traces/features_ll_64.csv")
    df98 = pd.read_csv("../mem_access_traces/features_ll_128.csv")
    df99 = pd.read_csv("../mem_access_traces/features_ll_256.csv")
    df100 = pd.read_csv("../mem_access_traces/features_ll_512.csv")
    df101 = pd.read_csv("../mem_access_traces/features_ll_1024.csv")

    ll_dfs = [df92,df93,df94,df95,df96,df97,df98,df99,df100,df101]
    ll_merged_df = pd.concat(ll_dfs, axis=0,ignore_index=True)
    ll_merged_df.to_csv("../mem_access_traces/ll_merged4.csv",index=False)







    all_dfs = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17,df18,df19,df20,df21,df22,df23,df24,df26,df26,df27,df28,df29,df30,df31,df32,df33,df34,df35,df36,df37,df38,df39,df40,df41,df42,df43,df44,df45,df46,df47,df48,df49,df50,df51,df52,df53,df54,df55,df56,df57,df58,df59,df60,df61,df62,df63,df64,df65,df66,df67,df68,df69,df70,df71,df72,df73,df74,df75,df76,df77,df78,df79,df80,df81,df82,df83,df84,df85,df86,df87,df88,df89,df90,df91,df92,df93,df94,df95,df96,df97,df98,df99,df100,df101]

    merged_df = pd.concat(all_dfs, axis=0,ignore_index=True)
    print(merged_df.shape)
    
    merged_df.to_csv("../mem_access_traces/merged4.csv",index=False)

add_target_col()
merge_dfs()
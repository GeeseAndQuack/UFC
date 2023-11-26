import pandas as pd
import typing
import matplotlib.pyplot as plt
import numpy as np

event_data = pd.read_csv("C:/Users/Jack/Documents/Coding Projects/UFC/Data/ufc_event_data.csv")
fight_data = pd.read_csv("C:/Users/Jack/Documents/Coding Projects/UFC/Data/ufc_fight_data.csv")
fight_stat_data = pd.read_csv("C:/Users/Jack/Documents/Coding Projects/UFC/Data/ufc_fight_stat_data.csv")
fighter_data = pd.read_csv("C:/Users/Jack/Documents/Coding Projects/UFC/Data/ufc_fighter_data.csv")

def stat_prcnt_calc(df: pd.DataFrame, att_var: str, succ_var: str) -> pd.Series:
    return df[succ_var]/df[att_var]

def row_sum_ignore_na(row: pd.Series, cols: list) -> float:

    if all([pd.isna(row[i]) for i in cols]): # If all values are na return na
        return np.nan

    res = 0
    for i in cols:
        if pd.isna(row[i]) == False:
            res += row[i]
    return res

df = fight_stat_data.merge(fighter_data, 
    how = "left", 
    left_on = "fighter_id", 
    right_on = "fighter_id")

print(df.size)

df["strike_succ_prcnt"] = stat_prcnt_calc(df,"total_strikes_att", "total_strikes_succ")
df["sig_strike_succ_prcnt"] = stat_prcnt_calc(df,"sig_strikes_att", "sig_strikes_succ")
df["takedown_succ_prct"] = stat_prcnt_calc(df,"takedown_att", "takedown_succ")

df["tot_fights"] = df.apply(
    lambda row: row_sum_ignore_na(
        row, ["fighter_w", "fighter_l", "fighter_d", "fighter_nc_dq"]),
         axis = 1)

tot_ufc_fights = df.groupby("fighter_id").size().reset_index(name="Count")

tot_ufc_fights = tot_ufc_fights.merge(fighter_data[["fighter_id","fighter_f_name", "fighter_l_name"]], how = "left", left_on = "fighter_id", right_on = "fighter_id")

df.dropna(subset = ["fighter_f_name", "fighter_l_name"], inplace = True)

df.size
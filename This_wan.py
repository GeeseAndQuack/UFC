import pandas as pd
import typing
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing

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

def min_sec_to_sec(row: pd.Series, time: str) -> float:
    try:
        mins_secs = row[time].split(":")
    except:
        return np.nan
    else:
        if len(mins_secs) == 2:
            mins = int(mins_secs[0]) * 60
            secs = int(mins_secs[1])
            res = mins + secs
            return res
        else:
            return np.nan
        
df = fight_stat_data.merge(fighter_data, 
    how = "left", 
    left_on = "fighter_id", 
    right_on = "fighter_id")

df["strike_succ_prcnt"] = stat_prcnt_calc(df,"total_strikes_att", "total_strikes_succ")
df["sig_strike_succ_prcnt"] = stat_prcnt_calc(df,"sig_strikes_att", "sig_strikes_succ")
df["takedown_succ_prcnt"] = stat_prcnt_calc(df,"takedown_att", "takedown_succ")

# Calculating total number of fights - using row_sum_ignore_na
df["tot_fights"] = df.apply(
    lambda row: row_sum_ignore_na(
        row, ["fighter_w", "fighter_l", "fighter_d", "fighter_nc_dq"]),
         axis = 1)

# Calculating total ufc fights by counting rows when grouping by fighter_id
tot_ufc_fights = df.groupby("fighter_id").size().reset_index(name="Count")
# Joining with fighter_data to get name for fighter
tot_ufc_fights = tot_ufc_fights.merge(fighter_data[["fighter_id","fighter_f_name", "fighter_l_name"]], how = "left", left_on = "fighter_id", right_on = "fighter_id")

# Raw winner variable is as the fighter_id, want to conver this to a binary when joint on fighter_data
wins = fight_data[["fight_id", 
                  "event_id", 
                  "winner", 
                  "result", 
                  "result_details", 
                  "finish_round", 
                  "finish_time"]]

# Creating dependent variable by merging the above dataframe
df = df.merge(wins, how = "left", left_on = "fight_id", right_on = "fight_id")
df["won"] = df.apply(lambda row: 1 if row["fighter_id"] == row["winner"] else 0, axis = 1)
df.drop("winner", inplace = True, axis = 1) # We have replaced this with our binary "won"

# Converting ctrl_time and finish_time into a integer representation as seconds
df["ctrl_time_sec"] = df.apply(
    lambda row: min_sec_to_sec(row, "ctrl_time"), axis = 1
)
df["finish_time_sec"] = df.apply(
    lambda row: min_sec_to_sec(row, "finish_time"), axis = 1
)

df["result"] = df["result"].astype("category")

enc = preprocessing.OneHotEncoder(sparse_output=False)
enc_data = enc.fit_transform(df[["result"]])
enc_feature_names = enc.get_feature_names_out(["result"])

df1 = pd.concat([df, pd.DataFrame(enc_data, columns = enc_feature_names)], axis = 1)

df1_a = df1[df1["fight_id"].duplicated(keep="first")]
df1_b = df1[df1["fight_id"].duplicated(keep="last")]

vs_df = pd.merge(df1_a, df1_a, how="left", on="fight_id", suffixes=["_A", "_B"])

xs = df1.drop(["fighter_url",
            "fight_url",
            "fighter_nickname", 
            "event_id", 
            "fight_stat_id", 
            "fight_id", 
            "fighter_id",
            "won",
            "fighter_f_name",
            "fighter_l_name",
            "result",
            "result_details",
            "finish_time",
            "ctrl_time"], 
            axis = 1)

xs = xs.apply(pd.to_numeric, errors = "coerce")
xs = (xs-xs.mean())/xs.std()
xs = xs.fillna(0)

y = df["won"]

xs = xs.fillna(0)

vaz = ["fighter_url",
            "fight_url",
            "fighter_nickname", 
            "event_id", 
            "fight_stat_id", 
            "fight_id", 
            "fighter_id",
            "won",
            "fighter_f_name",
            "fighter_l_name",
            "result",
            "result_details",
            "finish_time",
            "ctrl_time"]

a_vars = [i + "_A" for i in vaz if i != "fight_id"]
b_vars = [i + "_B" for i in vaz if i != "fight_id" and "won_B"]
ab_vars = a_vars + b_vars + ["fight_id"]

xs = vs_df.drop(ab_vars, axis = 1)

xs = xs.apply(pd.to_numeric, errors = "coerce")
xs = (xs-xs.mean())/xs.std()
xs = xs.fillna(0)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    xs, y, test_size = 0.33, random_state=5)

# Modelling
model = linear_model.LogisticRegression(random_state=1,
                                        max_iter = 500,
                                        penalty = "elasticnet",
                                        solver = "saga",
                                        l1_ratio = 0.5 
                                        ).fit(X_train, y_train)

preds = model.predict(X_test)
model.score(X_test, y_test)

# Think Bayesian makes more sense

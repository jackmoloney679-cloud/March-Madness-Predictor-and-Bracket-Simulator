#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


try:
    Seeds = pd.read_csv('data/raw/MNCAATourneySeeds.csv')
    Conferences = pd.read_csv('data/raw/MTeamConferences.csv')
    RegularSeasonDetail = pd.read_csv('data/raw/MRegularSeasonDetailedResults.csv')
    TourneyCompact = pd.read_csv('data/raw/MNCAATourneyCompactResults.csv')

    print("✅ Data loaded successfully.")
    print(RegularSeasonDetail.columns.values)

except FileNotFoundError as e:
    print("⚠️ Data files not found.")
    print(str(e))
    print("\n➡️ To run the full pipeline, place the Kaggle CSV files in:")
    print("   data/raw/")
    print("\n✅ main.py executed successfully (no crash).")
    raise SystemExit(0)


# In[13]:


WinningTeams = pd.DataFrame()
LosingTeams = pd.DataFrame()

Columns = ['Season', 'TeamID', 'Score', 'OppPoints',
       'Loc', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppFGM', 'OppFGA',
       'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO',
       'OppStl', 'OppBlk', 'OppPF']

WinningTeams[Columns] = RegularSeasonDetail[['Season', 'WTeamID', 'WScore','LScore',
       'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
       'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA',
       'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
       'LStl', 'LBlk', 'LPF']]

WinningTeams['Win'] = 1
WinningTeams['Loss'] = 0


LosingTeams[Columns] = RegularSeasonDetail[['Season', 'LTeamID', 'LScore','WScore',
       'WLoc', 'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
       'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA',
       'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
       'WStl', 'WBlk', 'WPF']]

def change_loc(loc):
    if loc == 'H':
        return 'A'
    elif loc == 'A':
        return 'H'
    else :
        return 'N'
    
LosingTeams['Loc'] = LosingTeams['Loc'].apply(change_loc)

LosingTeams['Win'] = 0
LosingTeams['Loss'] = 1

WinningLosingTeams = pd.concat([WinningTeams, LosingTeams])
CombinedTeams = WinningLosingTeams.groupby(['Season', 'TeamID' ]).sum()
CombinedTeams['NumGames'] = CombinedTeams['Win'] + CombinedTeams['Loss'] 
print(CombinedTeams.columns.values)


# In[14]:


RegularSeasonInput = pd.DataFrame()

RegularSeasonInput['WinRatio'] = CombinedTeams['Win'] / CombinedTeams['NumGames']
RegularSeasonInput['PointsPerGame'] = CombinedTeams['Score'] / CombinedTeams['NumGames']
RegularSeasonInput['PointsAllowedPerGame'] = CombinedTeams['OppPoints'] / CombinedTeams['NumGames']
RegularSeasonInput['PointsRatio'] = CombinedTeams['Score'] / CombinedTeams['OppPoints']
RegularSeasonInput['OTPerGame'] = CombinedTeams['NumOT'] / CombinedTeams['NumGames']

RegularSeasonInput['FGPerGame'] = CombinedTeams['FGM'] / CombinedTeams['NumGames']
RegularSeasonInput['FGRatio'] = CombinedTeams['FGM'] / CombinedTeams['FGA']
RegularSeasonInput['FGAllowedPerGame'] = CombinedTeams['OppFGM'] / CombinedTeams['NumGames']

RegularSeasonInput['FG3PerGame'] = CombinedTeams['FGM3'] / CombinedTeams['NumGames']
RegularSeasonInput['FG3Ratio'] = CombinedTeams['FGM3'] / CombinedTeams['FGA3']
RegularSeasonInput['FG3AllowedPerGame'] = CombinedTeams['OppFGM3'] / CombinedTeams['NumGames']

RegularSeasonInput['FTPerGame'] = CombinedTeams['FTM'] / CombinedTeams['NumGames']
RegularSeasonInput['FTRatio'] = CombinedTeams['FTM'] / CombinedTeams['FTA']
RegularSeasonInput['FTAllowedPerGame'] = CombinedTeams['OppFTM'] / CombinedTeams['NumGames']

RegularSeasonInput['ORRatio'] = CombinedTeams['OR'] / (CombinedTeams['OR'] + CombinedTeams['OppDR'])
RegularSeasonInput['DRRatio'] = CombinedTeams['DR'] / (CombinedTeams['DR'] + CombinedTeams['OppOR'])

RegularSeasonInput['AstPerGame'] = CombinedTeams['Ast'] / CombinedTeams['NumGames']

RegularSeasonInput['TOPerGame'] = CombinedTeams['TO'] / CombinedTeams['NumGames']
RegularSeasonInput['StlPerGame'] = CombinedTeams['Stl'] / CombinedTeams['NumGames']
RegularSeasonInput['BlkPerGame'] = CombinedTeams['Blk'] / CombinedTeams['NumGames']
RegularSeasonInput['PFPerGame'] = CombinedTeams['PF'] / CombinedTeams['NumGames']

print(RegularSeasonInput)


# In[15]:


Seed_dict = Seeds.set_index(['Season', 'TeamID'])

TournamentInput = pd.DataFrame()

WinIDs = TourneyCompact['WTeamID']
LoseIDs = TourneyCompact['LTeamID']
Season = TourneyCompact['Season']

Winners = pd.DataFrame()
Winners[['Season','Team1','Team2']] = TourneyCompact[['Season','WTeamID','LTeamID']]
Winners['Result'] = 1

Losers = pd.DataFrame()
Losers[['Season','Team1','Team2']] = TourneyCompact[['Season','LTeamID','WTeamID']]
Losers['Result'] = 0

TournamentInput = pd.concat([Winners,Losers])
TournamentInput = TournamentInput[TournamentInput['Season']>=2003].reset_index(drop=True)

Team1Seeds = []
Team2Seeds = [] 

for x in range(len(TournamentInput)):
    index = (TournamentInput['Season'][x],TournamentInput['Team1'][x])
    Seed = Seed_dict.loc[index].values[0]
    if len(Seed) == 4:
        Seed = int(Seed[1:-1])
    else:
        Seed = int(Seed[1:])
    Team1Seeds.append(Seed)
    
    index = (TournamentInput['Season'][x],TournamentInput['Team2'][x])
    Seed = Seed_dict.loc[index].values[0]
    if len(Seed) == 4:
         Seed = int(Seed[1:-1])
    else:
        Seed = int(Seed[1:])
    Team2Seeds.append(Seed)

TournamentInput['Team1Seeds']= Team1Seeds
TournamentInput['Team2Seeds']= Team2Seeds
print(TournamentInput)


# In[16]:


outscores = []

for x in range(len(TournamentInput)):
    index = (TournamentInput['Season'][x],TournamentInput['Team1'][x])
    Team1Score = RegularSeasonInput.loc[index]
    Team1Score['Seed'] = TournamentInput['Team1Seeds'][x]
    
    index = (TournamentInput['Season'][x],TournamentInput['Team2'][x])
    Team2Score = RegularSeasonInput.loc[index]
    Team2Score['Seed'] = TournamentInput['Team2Seeds'][x]
    
    outscore = Team1Score - Team2Score
    outscore['Result'] = TournamentInput['Result'][x]
    outscores.append(outscore)
    
outscores = pd.DataFrame(outscores)
print(outscores)
print(outscores.describe())


# In[17]:


Correlations = round(outscores.corr(), 2)
print(np.abs(Correlations['Result']))

import seaborn as sns 
plt.figure(figsize=(15,10))

sns.heatmap(Correlations)
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# In[18]:


x = outscores[outscores.columns[:-1].values]
y = outscores['Result'].values

np.random.seed(1)
index = np.random.permutation(len(x))
Train_index = index[:int(-.2*len(x))]
Test_index = index[int(-.2*len(x)):]

x_Train = x.iloc[Train_index]
x_Test = x.iloc[Test_index]
y_Train = y[Train_index]
y_Test = y[Test_index]

mins = x_Train.min(axis=0)
maxs = x_Train.max(axis=0)

x_Train = (x_Train - mins)/(maxs - mins)
x_Test = (x_Test - mins)/(maxs - mins)

print(x_Train.shape, x_Test.shape, y_Train.shape, y_Test.shape)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

def train_tournament_model(outscores: pd.DataFrame):
    
    X = outscores[outscores.columns[:-1]]
    y = outscores['Result'].values

    np.random.seed(1)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)

    X_train = (X_train - mins) / (maxs - mins)
    X_test = (X_test - mins) / (maxs - mins)

    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Brier Score:", round(brier_score_loss(y_test, y_proba), 3))

    return model, mins, maxs, X_test, y_test


# In[20]:


model, mins, maxs, X_test, y_test = train_tournament_model(outscores)


# In[21]:


def parse_seed(seed_str : str) -> int:
    
    if len(seed_str) == 4:
        return int(seed_str)[1:-1]
    
    else : 
        return int(seed_str[1:])


# In[22]:


def build_matchup_features(season: int,
                           team1ID: int,
                           team2ID: int, 
                           regular_season_input: pd.DataFrame,
                           seeds_df: pd.DataFrame) -> pd.Series:

    seed_lookup = seeds_df.set_index(['Season', 'TeamID'])

    team1stats = regular_season_input.loc[(season, team1ID)].copy()
    team2stats = regular_season_input.loc[(season, team2ID)].copy()

    raw_seed1 = seed_lookup.loc[(season, team1ID), 'Seed']
    raw_seed2 = seed_lookup.loc[(season, team2ID), 'Seed']

    def parse_seed_safe(seed_str):
        if pd.isna(seed_str):
            return 0
        digits = "".join(filter(str.isdigit, seed_str))
        if digits == "":
            return 0
        return int(digits)

    team1stats['Seed'] = parse_seed_safe(raw_seed1)
    team2stats['Seed'] = parse_seed_safe(raw_seed2)

    return team1stats - team2stats


# In[23]:


row = TournamentInput.iloc[0]

season = row['Season']
team1 = row['Team1']
team2 = row['Team2']

print('Season:', season, 'Team1:', team1, 'Team2:', team2)

features = build_matchup_features(season, team1, team2, RegularSeasonInput, Seeds)
aligned = features.reindex(mins.index, fill_value=0)
scaled = (aligned - mins) / (maxs - mins)

print(features)
print('Number of features:', len(features))


# In[24]:


X_input = pd.DataFrame(scaled.values.reshape(1, -1), columns=mins.index)

features = build_matchup_features(season, team1, team2, RegularSeasonInput, Seeds)
aligned = features.reindex(mins.index, fill_value=0)
scaled = (aligned - mins) / (maxs - mins)


prob_team1_wins = model.predict_proba(X_input)[0, 1]

print("Probability that Team 1 wins:", round(prob_team1_wins, 3))


# In[25]:


def predict_winner(season, team1, team2, model, mins, maxs, RegularSeasonInput, Seeds):

    features = build_matchup_features(season, team1, team2, RegularSeasonInput, Seeds)
    aligned = features.reindex(mins.index, fill_value=0)
    scaled = (aligned - mins) / (maxs - mins)

    X_input = pd.DataFrame(scaled.values.reshape(1, -1), columns=mins.index)
    proba_team1 = model.predict_proba(X_input)[0, 1]

    return 1 if proba_team1 > 0.5 else 0


# In[26]:


winner = predict_winner(2003, 1421, 1411, model, mins, maxs, RegularSeasonInput, Seeds)

print('Predicted winner (1 means Team1 wins):', winner)


# In[27]:


EPS = 1e-6

def predict_game_probability(season, team1, team2, model, mins, maxs, regular_season_input, seeds_df):
    features = build_matchup_features(season, team1, team2, regular_season_input, seeds_df)

    aligned = features.reindex(mins.index, fill_value=0)

    scaled = (aligned - mins) / (maxs - mins)
    scaled = scaled.fillna(0)

    X_input = pd.DataFrame([scaled.values], columns=mins.index)

    p = model.predict_proba(X_input)[0, 1]

    return float(np.clip(p, EPS, 1 - EPS))


# In[28]:


def proba_to_decimal_odds(p):
    return 1.0 / p

def build_model_odds_for_season(season, games_df, model, mins, maxs, regular_season_input, seeds_df):
    rows = []
    for _, r in games_df.iterrows():
        t1 = int(r["Team1"])
        t2 = int(r["Team2"])
        true = int(r["Result"]) if "Result" in r else int(r["TrueResult"])

        p = predict_game_probability(season, t1, t2, model, mins, maxs, regular_season_input, seeds_df)

        rows.append({
            "Season": season,
            "Team1": t1,
            "Team2": t2,
            "TrueResult": true,
            "WinProba": p,
            "DecimalOdds": proba_to_decimal_odds(p),
        })
    return pd.DataFrame(rows)


# In[29]:


from sklearn.metrics import brier_score_loss, log_loss

model_odds_2003 = build_model_odds_for_season(2003, TournamentInput[TournamentInput["Season"]==2003], 
                                             model, mins, maxs, RegularSeasonInput, Seeds)

y = model_odds_2003["TrueResult"].values
p = model_odds_2003["WinProba"].values

print("Games:", len(model_odds_2003))
print("Brier:", brier_score_loss(y, p))
print("LogLoss:", log_loss(y, p))
print("Accuracy:", ((p >= 0.5).astype(int) == y).mean())

print(model_odds_2003.head())


# In[30]:


from sklearn.metrics import brier_score_loss, log_loss

def evaluate_odds_df(df):
    y = df["TrueResult"].values
    p = df["WinProba"].values
    return {
        "games": len(df),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
        "accuracy": ((p >= 0.5).astype(int) == y).mean(),
        "mean_p": float(np.mean(p)),
    }


# In[31]:


seasons = sorted(TournamentInput["Season"].unique())

all_odds = []
season_metrics = []

for season in seasons:
    season_games = TournamentInput[TournamentInput["Season"] == season].copy()

    df_odds = build_model_odds_for_season(
        season, season_games,
        model, mins, maxs,
        RegularSeasonInput, Seeds
    )

    all_odds.append(df_odds)

    m = evaluate_odds_df(df_odds)
    m["Season"] = season
    season_metrics.append(m)

model_odds_all = pd.concat(all_odds, ignore_index=True)
metrics_by_season = pd.DataFrame(season_metrics).sort_values("Season")

print(metrics_by_season.head())
print(metrics_by_season.tail())
print("Total rows in model_odds_all:", len(model_odds_all))


# In[32]:


model_odds_all.to_csv("model_odds_all_seasons.csv", index=False)
metrics_by_season.to_csv("model_metrics_by_season.csv", index=False)

print("Saved:")
print("- model_odds_all_seasons.csv")
print("- model_metrics_by_season.csv")


# In[46]:


plt.figure(figsize=(10,4))
plt.plot(metrics_by_season["Season"], metrics_by_season["brier"])
plt.xlabel("Season")
plt.ylabel("Brier score (lower is better)")
plt.title("Model probability quality over time")
plt.savefig("Model_probability_quality_over_time.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(metrics_by_season["Season"], metrics_by_season["accuracy"])
plt.xlabel("Season")
plt.ylabel("Accuracy")
plt.title("Model accuracy over time")
plt.savefig("Model_accuracy_over_time.png", dpi=300, bbox_inches="tight")
plt.show()


# In[34]:


from sklearn.linear_model import LogisticRegression


seed_lookup = Seeds.set_index(["Season", "TeamID"])["Seed"]

def seed_to_int(seed_str):
    
    digits = "".join([c for c in str(seed_str) if c.isdigit()])
    return int(digits) if digits else np.nan

def add_seed_diff(df):
    s1 = df.apply(lambda r: seed_to_int(seed_lookup.loc[(r["Season"], r["Team1"])]), axis=1)
    s2 = df.apply(lambda r: seed_to_int(seed_lookup.loc[(r["Season"], r["Team2"])]), axis=1)
    out = df.copy()
    out["Seed1"] = s1
    out["Seed2"] = s2
    out["SeedDiff"] = out["Seed1"] - out["Seed2"]
    return out.dropna(subset=["SeedDiff", "Result"])

baseline_df = add_seed_diff(TournamentInput.rename(columns={"TrueResult":"Result"} if "TrueResult" in TournamentInput.columns else {}))

X_seed = baseline_df[["SeedDiff"]].values
y_seed = baseline_df["Result"].values

seed_model = LogisticRegression()
seed_model.fit(X_seed, y_seed)

print("Seed-only baseline fitted.")


# In[35]:


temp = model_odds_all.merge(
    baseline_df[["Season", "Team1", "Team2", "SeedDiff"]],
    on=["Season", "Team1", "Team2"],
    how="left"
)

temp["BaselineProba"] = seed_model.predict_proba(temp[["SeedDiff"]].values)[:, 1]

model_vs_baseline = temp.dropna(subset=["BaselineProba"]).copy()
print("Rows with baseline:", len(model_vs_baseline), "out of", len(model_odds_all))


# In[36]:


from sklearn.metrics import brier_score_loss, log_loss

y = model_vs_baseline["TrueResult"].values
p_model = model_vs_baseline["WinProba"].values
p_base  = model_vs_baseline["BaselineProba"].values

print("OVERALL (2003–2024)")
print("Model    Brier:", brier_score_loss(y, p_model), " LogLoss:", log_loss(y, p_model))
print("Baseline Brier:", brier_score_loss(y, p_base),  " LogLoss:", log_loss(y, p_base))


rows = []
for season, g in model_vs_baseline.groupby("Season"):
    y_s = g["TrueResult"].values
    pm = g["WinProba"].values
    pb = g["BaselineProba"].values
    rows.append({
        "Season": int(season),
        "games": len(g),
        "brier_model": brier_score_loss(y_s, pm),
        "brier_baseline": brier_score_loss(y_s, pb),
        "logloss_model": log_loss(y_s, pm),
        "logloss_baseline": log_loss(y_s, pb),
    })

compare_by_season = pd.DataFrame(rows).sort_values("Season")
print(compare_by_season.head())
print(compare_by_season.tail())


# In[37]:


model_vs_baseline["Gap"] = model_vs_baseline["WinProba"] - model_vs_baseline["BaselineProba"]

top = model_vs_baseline.reindex(model_vs_baseline["Gap"].abs().sort_values(ascending=False).index).head(20)

print(top[["Season","Team1","Team2","TrueResult","WinProba","BaselineProba","Gap","SeedDiff"]])


# In[38]:


def explain_matchup(season, team1, team2):
    feats = build_matchup_features(int(season), int(team1), int(team2), RegularSeasonInput, Seeds)
    s = feats.drop(labels=[c for c in feats.index if c.lower()=="result"], errors="ignore")
    return s.reindex(s.abs().sort_values(ascending=False).index)

example = top.iloc[0]
season, t1, t2 = example["Season"], example["Team1"], example["Team2"]

print("Example biggest gap:")
print("Season", season, "Team1", t1, "Team2", t2)
print("Model:", round(example["WinProba"],3), "Baseline:", round(example["BaselineProba"],3), "True:", int(example["TrueResult"]))
print(explain_matchup(season, t1, t2).head(10))


# In[39]:


compare_by_season.to_csv("model_vs_baseline_by_season.csv", index=False)


# In[40]:


plt.figure(figsize=(10,4))
plt.plot(compare_by_season["Season"], compare_by_season["brier_model"], label="Model")
plt.plot(compare_by_season["Season"], compare_by_season["brier_baseline"], label="Baseline")
plt.xlabel("Season")
plt.ylabel("Brier Score")
plt.title("Probability Quality: Model vs Seed Baseline")
plt.legend()
plt.savefig("probability_quality_model_vs_baseline.png", dpi=300, bbox_inches="tight")
plt.show()


# In[41]:


Slots = pd.read_csv("data/raw/MNCAATourneySlots.csv")      
Teams = pd.read_csv("data/raw/MTeams.csv")                

print(Slots.head())
print(Teams.head())


# In[42]:


def seed_to_int(seed_str: str) -> int:
   
    digits = "".join([c for c in str(seed_str) if c.isdigit()])
    return int(digits) if digits else -1

team_name = Teams.set_index("TeamID")["TeamName"].to_dict()


# In[43]:


def make_seed_team_map(season: int, seeds_df: pd.DataFrame) -> dict:
    s = seeds_df[seeds_df["Season"] == season].copy()
    return dict(zip(s["Seed"].astype(str), s["TeamID"].astype(int)))

season_to_run = 2024
seed_team = make_seed_team_map(season_to_run, Seeds)

print("Seeds loaded for season:", season_to_run, "count:", len(seed_team))
list(seed_team.items())[:5]


# In[41]:


import random

def pick_winner(season, team1, team2, model, mins, maxs, regular_season_input, seeds_df, mode="deterministic"):
    p = predict_game_probability(season, team1, team2, model, mins, maxs, regular_season_input, seeds_df)
    if mode == "random":
        return team1 if random.random() < p else team2
    
    return team1 if p >= 0.5 else team2


# In[42]:


def simulate_bracket_from_slots(season: int,
                                seeds_df: pd.DataFrame,
                                slots_df: pd.DataFrame,
                                model, mins, maxs,
                                regular_season_input: pd.DataFrame,
                                mode="deterministic"):
    slot_winner = {}

    seed_team = make_seed_team_map(season, seeds_df)

    S = slots_df[slots_df["Season"] == season].copy()

    unresolved = True
    rounds_log = []

    while unresolved:
        unresolved = False
        progress = False

        for _, row in S.iterrows():
            slot = row["Slot"]
            strong = row["StrongSeed"]
            weak = row["WeakSeed"]

            
            if slot in slot_winner:
                continue

            
            if strong in seed_team:
                t1 = seed_team[strong]
            elif strong in slot_winner:
                t1 = slot_winner[strong]
            else:
                unresolved = True
                continue

            
            if weak in seed_team:
                t2 = seed_team[weak]
            elif weak in slot_winner:
                t2 = slot_winner[weak]
            else:
                unresolved = True
                continue

            
            winner = pick_winner(season, t1, t2, model, mins, maxs, RegularSeasonInput, Seeds, mode=mode)
            slot_winner[slot] = winner
            progress = True

            rounds_log.append({
                "Season": season,
                "Slot": slot,
                "Strong": strong,
                "Weak": weak,
                "Team1": t1,
                "Team2": t2,
                "Winner": winner
            })

        if unresolved and not progress:
            
            missing = set(S["StrongSeed"]).union(set(S["WeakSeed"])) - set(seed_team.keys()) - set(slot_winner.keys())
            raise ValueError(f"Could not resolve bracket. Missing references: {sorted(list(missing))[:20]}")

    games_df = pd.DataFrame(rounds_log)

    
    champion_slot = "R6CH" if "R6CH" in slot_winner else list(slot_winner.keys())[-1]
    champion_id = slot_winner[champion_slot]

    return games_df, champion_id, champion_slot

games_2024, champ_id, champ_slot = simulate_bracket_from_slots(
    season_to_run, Seeds, Slots,
    model, mins, maxs, RegularSeasonInput,
    mode="deterministic"   
)

print("Champion slot:", champ_slot)
print("Champion TeamID:", champ_id, "-", team_name.get(champ_id, "Unknown"))
print(games_2024.tail(10))


# In[43]:


def add_names(df):
    df = df.copy()
    df["Team1Name"] = df["Team1"].map(team_name)
    df["Team2Name"] = df["Team2"].map(team_name)
    df["WinnerName"] = df["Winner"].map(team_name)
    return df

print(add_names(games_2024).tail(15)[["Slot","Team1Name","Team2Name","WinnerName"]])


# In[44]:


def print_bracket(games_df, team_name_map):
    rounds = {
        "R1": "Round of 64",
        "R2": "Round of 32",
        "R3": "Sweet 16",
        "R4": "Elite 8",
        "R5": "Final Four",
        "R6": " National Championship"
    }

    for r_code, r_name in rounds.items():
        print("\n" + "="*len(r_name))
        print(r_name)
        print("="*len(r_name))

        g = games_df[games_df["Slot"].str.startswith(r_code)]
        for _, row in g.iterrows():
            t1 = team_name_map.get(row["Team1"], row["Team1"])
            t2 = team_name_map.get(row["Team2"], row["Team2"])
            w  = team_name_map.get(row["Winner"], row["Winner"])
            print(f"{t1} vs {t2}  →  {w}")

print_bracket(games_2024, team_name)


# In[83]:


from collections import Counter

def monte_carlo_champions(season, n_sims=500):
    champs = []
    for _ in range(n_sims):
        _, champ_id, _ = simulate_bracket_from_slots(
            season, Seeds, Slots,
            model, mins, maxs, RegularSeasonInput,
            mode="random"
        )
        champs.append(champ_id)
    return Counter(champs)

champ_counts = monte_carlo_champions(2024, n_sims=500)

champ_df = (
    pd.DataFrame(champ_counts.items(), columns=["TeamID","Count"])
      .assign(Probability=lambda d: d["Count"]/d["Count"].sum())
      .sort_values("Probability", ascending=False)
)

champ_df["TeamName"] = champ_df["TeamID"].map(team_name)
print(champ_df.head(10))


# In[45]:


import re

def seed_to_int(seed_str: str) -> int:
    # Handles 'W01', 'X16a', etc.
    digits = "".join([c for c in str(seed_str) if c.isdigit()])
    return int(digits) if digits else -1

def make_seed_map_for_season(season: int, seeds_df):
    s = seeds_df[seeds_df["Season"] == season].copy()
    return {int(tid): seed_to_int(seed) for tid, seed in zip(s["TeamID"], s["Seed"])}

LABEL_BOX = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85)


# In[46]:


import re

PAIR_ORDER = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
PAIR_RANK = {p:i for i,p in enumerate(PAIR_ORDER)}

def _seednum(seed_str: str) -> int:
    # works for 'W01', 'X16a', etc.
    digits = "".join(ch for ch in str(seed_str) if ch.isdigit())
    return int(digits) if digits else -1

def build_region_slot_order(season: int, region: str, slots_df: pd.DataFrame) -> dict:
    
    region = region.upper()
    S = slots_df[slots_df["Season"] == season].copy()
    S = S[S["Slot"].astype(str).str.match(rf"^R[1-4]{region}")].copy()

    slot_order = {}

    
    r1 = S[S["Slot"].astype(str).str.startswith("R1")].copy()
    for _, r in r1.iterrows():
        a = _seednum(r["StrongSeed"])
        b = _seednum(r["WeakSeed"])
        pair = (min(a,b), max(a,b))
        
        slot_order[str(r["Slot"])] = PAIR_RANK.get(pair, 999)

    
    for rnd in ["R2","R3","R4"]:
        rr = S[S["Slot"].astype(str).str.startswith(rnd)].copy()
        for _, r in rr.iterrows():
            child1 = str(r["StrongSeed"])
            child2 = str(r["WeakSeed"])
            slot_order[str(r["Slot"])] = min(slot_order.get(child1, 999), slot_order.get(child2, 999))

    return slot_order


# In[47]:


import matplotlib.pyplot as plt

def plot_region_bracket_fixed(games_df, team_name_map, seeds_df, season, region, savepath=None):
    region = region.upper()
    seed_map = make_seed_map_for_season(season, seeds_df)

    
    
    g = games_df[games_df["Slot"].astype(str).str.match(rf"^R[1-4]{region}")].copy()
    g["Round"] = g["Slot"].astype(str).str.extract(r"^(R\d)")


    slot_order = build_region_slot_order(season, region, Slots)
    g["Order"] = g["Slot"].astype(str).map(slot_order)


    g = g.sort_values(["Round", "Order", "Slot"]).reset_index(drop=True)


    r1 = g[g["Round"] == "R1"].reset_index(drop=True)
    r2 = g[g["Round"] == "R2"].reset_index(drop=True)
    r3 = g[g["Round"] == "R3"].reset_index(drop=True)
    r4 = g[g["Round"] == "R4"].reset_index(drop=True)

    counts = (len(r1), len(r2), len(r3), len(r4))
    if counts != (8, 4, 2, 1):
        print(f"Note: Region {region} counts (R1,R2,R3,R4) = {counts}. Still plotting what exists.")

    def label(tid: int) -> str:
        tid = int(tid)
        seed = seed_map.get(tid, -1)
        name = team_name_map.get(tid, str(tid))
        return f"({seed}) {name}" if seed != -1 else name

    # ---- Layout
    x_team = 0.0
    x_r1w  = 1.5
    x_r2w  = 3.5
    x_r3w  = 5.5
    x_r4w  = 7.5

    y_positions = list(range(15, -1, -1))  # 15..0

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.set_title(
    f"Predicted Bracket {season} — Region {region}",
    fontsize=16,
    fontweight="bold",
    pad=28   
)


    
    ax.text(x_team, 16.2, "Round of 64", fontsize=12, fontweight="bold")
    ax.text(x_r2w,  16.2, "Round of 32",   fontsize=12, fontweight="bold")
    ax.text(x_r3w,  16.2, "Sweet 16",   fontsize=12, fontweight="bold")
    ax.text(x_r4w,  16.2, "Elite 8",    fontsize=12, fontweight="bold")

    line_kw = dict(color="black", linewidth=1.4)

    
    r1_winner_y = []
    for i in range(min(len(r1), 8)):
        row = r1.iloc[i]
        t1, t2, w = int(row["Team1"]), int(row["Team2"]), int(row["Winner"])

        y1 = y_positions[2*i]
        y2 = y_positions[2*i + 1]
        yw = (y1 + y2) / 2
        r1_winner_y.append(yw)

        
        
        ax.text(x_team + 0.25, y1 + 0.15, label(t1),
        va="bottom", fontsize=10, bbox=LABEL_BOX)

        ax.text(x_team + 0.25, y2 + 0.15, label(t2),
        va="bottom", fontsize=10, bbox=LABEL_BOX)


        
        ax.plot([x_team + 0.2, x_team + 0.8], [y1, y1], **line_kw)
        ax.plot([x_team + 0.2, x_team + 0.8], [y2, y2], **line_kw)
        ax.plot([x_team + 0.8, x_r1w], [y1, yw], **line_kw)
        ax.plot([x_team + 0.8, x_r1w], [y2, yw], **line_kw)

        
        ax.text(x_r2w - 1.1, yw + 0.25, label(w), va="bottom",
                fontsize=10, fontweight="bold", bbox=LABEL_BOX)

    
    r2_winner_y = []
    for i in range(min(len(r2), 4)):
        w = int(r2.iloc[i]["Winner"])
        y_top = r1_winner_y[2*i]
        y_bot = r1_winner_y[2*i + 1]
        yw = (y_top + y_bot) / 2
        r2_winner_y.append(yw)

        ax.plot([x_r2w - 1.2, x_r2w - 0.6], [y_top, y_top], **line_kw)
        ax.plot([x_r2w - 1.2, x_r2w - 0.6], [y_bot, y_bot], **line_kw)
        ax.plot([x_r2w - 0.6, x_r2w], [y_top, yw], **line_kw)
        ax.plot([x_r2w - 0.6, x_r2w], [y_bot, yw], **line_kw)

        ax.text(x_r3w - 1.1, yw + 0.25, label(w), va="bottom",
                fontsize=10, fontweight="bold", bbox=LABEL_BOX)

    
    r3_winner_y = []
    for i in range(min(len(r3), 2)):
        w = int(r3.iloc[i]["Winner"])
        y_top = r2_winner_y[2*i]
        y_bot = r2_winner_y[2*i + 1]
        yw = (y_top + y_bot) / 2
        r3_winner_y.append(yw)

        ax.plot([x_r3w - 1.2, x_r3w - 0.6], [y_top, y_top], **line_kw)
        ax.plot([x_r3w - 1.2, x_r3w - 0.6], [y_bot, y_bot], **line_kw)
        ax.plot([x_r3w - 0.6, x_r3w], [y_top, yw], **line_kw)
        ax.plot([x_r3w - 0.6, x_r3w], [y_bot, yw], **line_kw)

        ax.text(x_r4w - 1.1, yw + 0.25, label(w), va="bottom",
                fontsize=11, fontweight="bold", bbox=LABEL_BOX)

    
    if len(r4) >= 1 and len(r3_winner_y) >= 2:
        w = int(r4.iloc[0]["Winner"])
        y_top, y_bot = r3_winner_y[0], r3_winner_y[1]
        yw = (y_top + y_bot) / 2

        ax.plot([x_r4w - 1.2, x_r4w - 0.6], [y_top, y_top], **line_kw)
        ax.plot([x_r4w - 1.2, x_r4w - 0.6], [y_bot, y_bot], **line_kw)
        ax.plot([x_r4w - 0.6, x_r4w], [y_top, yw], **line_kw)
        ax.plot([x_r4w - 0.6, x_r4w], [y_bot, yw], **line_kw)

        ax.text(x_r4w + 0.4, yw, f"Region {region} Winner:\n{label(w)}",
                va="center", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", alpha=0.9))

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# In[48]:


def plot_final_four_fixed(games_df, team_name_map, seeds_df, season, savepath=None):
    seed_map = make_seed_map_for_season(season, seeds_df)

    g = games_df[games_df["Slot"].astype(str).str.startswith(("R5","R6"))].copy()
    g["Round"] = g["Slot"].astype(str).str.extract(r"^(R\d)")
    g = g.sort_values(["Round","Slot"]).reset_index(drop=True)

    r5 = g[g["Round"] == "R5"].reset_index(drop=True)
    r6 = g[g["Round"] == "R6"].reset_index(drop=True)

    def label(tid: int) -> str:
        tid = int(tid)
        seed = seed_map.get(tid, -1)
        name = team_name_map.get(tid, str(tid))
        return f"({seed}) {name}" if seed != -1 else name

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_title(f"Predicted Bracket {season} — Final Four & Championship", fontsize=16, fontweight="bold")

    line_kw = dict(color="black", linewidth=1.6)

    
    x_team = 0.0
    x_mid  = 4.5
    x_ch   = 9.0
    x_label = x_team + 0.25

    y_top, y_bot = 3.5, 1.5

    ax.text(x_team, 4.6, "Final Four", fontsize=12, fontweight="bold")
    ax.text(x_mid,  4.6, "Championship", fontsize=12, fontweight="bold")

    if len(r5) >= 2:
        
        t1, t2, w1 = int(r5.iloc[0]["Team1"]), int(r5.iloc[0]["Team2"]), int(r5.iloc[0]["Winner"])
        ax.text(x_label, y_top+0.4 + 0.15, label(t1), va="bottom", fontsize=11, bbox=LABEL_BOX)
        ax.text(x_label, y_top-0.4 + 0.15, label(t2), va="bottom", fontsize=11, bbox=LABEL_BOX)
        ax.plot([x_team+0.2, x_mid-0.7], [y_top+0.4, y_top], **line_kw)
        ax.plot([x_team+0.2, x_mid-0.7], [y_top-0.4, y_top], **line_kw)
        ax.text(x_mid-0.5, y_top+0.25, label(w1), va="bottom",
                fontsize=12, fontweight="bold", bbox=LABEL_BOX)

        
        t3, t4, w2 = int(r5.iloc[1]["Team1"]), int(r5.iloc[1]["Team2"]), int(r5.iloc[1]["Winner"])
        ax.text(x_label, y_bot+0.4 + 0.15, label(t3), va="bottom", fontsize=11, bbox=LABEL_BOX)
        ax.text(x_label, y_bot-0.4 + 0.15, label(t4), va="bottom", fontsize=11, bbox=LABEL_BOX)
        ax.plot([x_team+0.2, x_mid-0.7], [y_bot+0.4, y_bot], **line_kw)
        ax.plot([x_team+0.2, x_mid-0.7], [y_bot-0.4, y_bot], **line_kw)
        ax.text(x_mid-0.5, y_bot+0.25, label(w2), va="bottom",
                fontsize=12, fontweight="bold", bbox=LABEL_BOX)

        
        ax.plot([x_mid, x_ch-0.8], [y_top, 2.5], **line_kw)
        ax.plot([x_mid, x_ch-0.8], [y_bot, 2.5], **line_kw)

    if len(r6) >= 1:
        champ = int(r6.iloc[0]["Winner"])
        ax.text(x_ch, 2.5, f"Champion:\n{label(champ)}",
                va="center", ha="center",
                fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="black", alpha=0.9))

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# In[49]:


season = 2024

for reg in ["W","X","Y","Z"]:
    plot_region_bracket_fixed(games_2024, team_name, Seeds, season, reg,
                              savepath=f"bracket_{season}_region_{reg}.png")

plot_final_four_fixed(games_2024, team_name, Seeds, season,
                      savepath=f"bracket_{season}_final_four.png")

print("Saved bracket images")



# In[ ]:





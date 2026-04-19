import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "combined_dataset_matching_features.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find the dataset at {DATA_PATH}.")

df = pd.read_csv(DATA_PATH)

print(df.shape)
print(df.columns.tolist())
df.head()

#Homophilly feature creation

df["age_diff"] = abs(df["Subject age"] - df["Partner age"])
df["same_gender"] = (df["Subject gender"] == df["Partner gender"]).astype(int)
df["same_race"] = 1 - df["Interracial relationship"]

print("Same race rate:")
print(df["same_race"].mean())

print("\nSame gender rate:")
print(df["same_gender"].mean())

print("\nAge difference:")
print(df["age_diff"].describe())

df["age_diff_bin"] = pd.cut(
    df["age_diff"],
    bins=[0, 0.25, 0.5, 1.0, 2.0, 10.0],
    labels=["very_small","small","medium","large","very_large"],
    include_lowest=True
)

df["age_diff_bin"].value_counts()

combo_counts = (
    df.groupby(["age_diff_bin","same_gender","same_race"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

combo_counts.head(10)

race_pairs = (
    df.groupby(["Subject race_WHITE","Partner race_WHITE"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

race_pairs.head(10)

pd.crosstab(df["same_race"], df["same_gender"])

subject_race_cols = [
    "Subject race_ASIAN",
    "Subject race_BLACK",
    "Subject race_HISPANIC",
    "Subject race_LATINO",
    "Subject race_NATIVE AMERICAN",
    "Subject race_OTHER",
    "Subject race_WHITE"
]

partner_race_cols = [
    "Partner race_ASIAN",
    "Partner race_BLACK",
    "Partner race_HISPANIC",
    "Partner race_LATINO",
    "Partner race_NATIVE AMERICAN",
    "Partner race_OTHER",
    "Partner race_WHITE"
]

df["Subject_race_label"] = (
    df[subject_race_cols]
    .idxmax(axis=1)
    .str.replace("Subject race_", "", regex=False)
)

df["Partner_race_label"] = (
    df[partner_race_cols]
    .idxmax(axis=1)
    .str.replace("Partner race_", "", regex=False)
)

race_pairs = (
    df.groupby(["Subject_race_label", "Partner_race_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(race_pairs)

race_pair_matrix = pd.crosstab(
    df["Subject_race_label"],
    df["Partner_race_label"]
)

print(race_pair_matrix)

race_pair_pct = pd.crosstab(
    df["Subject_race_label"],
    df["Partner_race_label"],
    normalize="all"
) * 100

race_pair_pct.round(2)

#plotting
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(race_pair_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Race Pair Combinations")
plt.xlabel("Partner Race")
plt.ylabel("Subject Race")
plt.show()
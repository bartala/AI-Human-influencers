import os
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

PTH = '...'

df = pd.read_csv(os.path.join(PTH,'AIVI_HI.csv'))
print(df['user_type'].value_counts())

"""# sample the data for manual checks"""

N_PER_GROUP = 140             # sample size per group
COL_TYPE = "user_type"  # "AIVI" / "HI"
COL_USER = "posts.comments.user"
COL_TEXT = "posts.comments.text"

# ------------------------------
# 1. Select one representative comment per user
# ------------------------------
# pick the LONGEST comment that a user wrote.
def pick_longest_comment(group):
    idx = group[COL_TEXT].astype(str).str.len().idxmax()
    return group.loc[idx]

unique_commenters = (
    df.groupby([COL_TYPE, COL_USER], as_index=False)
      .apply(pick_longest_comment, include_groups=False)
      .reset_index(drop=True)
)

# ------------------------------
# 2. Sample commenters per group
# ------------------------------
np.random.seed(42)

samples = []

for group_label in ["AI", "HUMAN"]:
    group_df = unique_commenters[unique_commenters[COL_TYPE] == group_label]
    n_available = len(group_df)
    n_sample = min(N_PER_GROUP, n_available)

    print(f"{group_label}: available = {n_available}, sampling = {n_sample}")

    sample_group = group_df.sample(
        n=n_sample, replace=False, random_state=42
    )
    samples.append(sample_group)

# Combine
sample_df = pd.concat(samples, ignore_index=True)

# ------------------------------
# 3. Add Instagram Profile URL
# ------------------------------
sample_df["profile_url"] = (
    "https://www.instagram.com/" + sample_df[COL_USER].astype(str) + "/"
)

# ------------------------------
# 4. Add empty column for manual coding
# ------------------------------
sample_df["is_bot_like"] = ""   # coder fills manually (1 or 0)

# ------------------------------
# 5. Save
# ------------------------------
sample_df.to_csv(os.path.join(PTH,"sampled_commenters_for_manual_coding.csv"), index=False)

"""# analyze the manual labels"""

# the code is expecting manual labels ('1' -> bot; '0'-> not a bot) in the "bot_like_label" column.

df = pd.read_csv(os.path.join(PTH,"sampled_commenters_for_manual_coding.csv"))

COL_INFLUENCER_TYPE = "user_type"

# Summary per group
summary = (
    df.groupby(COL_INFLUENCER_TYPE)["is_bot_like"]
      .agg(["sum", "count"])
      .rename(columns={"sum": "n_bot_like", "count": "n_total"})
)
summary["prop_bot_like"] = summary["n_bot_like"] / summary["n_total"]

print("Bot-like commenters by group:")
print(summary)

# Prepare 2x2 contingency table (AIVI vs HI x bot_like vs not_bot_like)
contingency = []

for group_label in ["AI", "HUMAN"]:
    subset = df[df[COL_INFLUENCER_TYPE] == group_label]
    n_bot = subset["is_bot_like"].sum()
    n_not = len(subset) - n_bot
    contingency.append([n_bot, n_not])

contingency

chi2, p, dof, expected = chi2_contingency(contingency)
print("\nChi-square test:")
print("chi2 =", chi2)
print("p-value =", p)
print("degrees of freedom =", dof)
print("expected counts =\n", expected)

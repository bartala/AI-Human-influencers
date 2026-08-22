"""
Figure 4: characteristics of comments classified as negative.

(a) account involvement (@mentions), negative vs non-negative, both groups;
(b) artificiality-term prevalence among negative comments, AIVI vs HI;
(c) artificiality-term prevalence within AIVIs, negative vs non-negative.

Run after mention_analysis.py and artificiality_keywords.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

AIVI_C, HI_C = "#3B6EA8", "#C4622D"
MIN_NEG = 30

art = pd.read_csv("output/artificiality_by_influencer.csv")
art_w = pd.read_csv("output/artificiality_within_aivi.csv")
men = pd.read_csv("output/mention_analysis_by_influencer.csv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
rng = np.random.default_rng(42)


def jitter(n, w=0.055):
    return rng.uniform(-w, w, n)


# ---- panel a: account involvement ---------------------------------------
ax = axes[0]
m = men[men.n_neg >= MIN_NEG]
groups, x, w = ["AIVI", "HI"], np.arange(2), 0.34
non_v = [100 * m[m.group == g]["mentions_other_nonneg"].median() for g in groups]
neg_v = [100 * m[m.group == g]["mentions_other_neg"].median() for g in groups]
ax.bar(x - w / 2, non_v, w, color="#C9C9C9", label="non-negative comments")
ax.bar(x + w / 2, neg_v, w, color="#6B6B6B", label="negative comments")
for i, g in enumerate(groups):
    for vals, off in [(m[m.group == g]["mentions_other_nonneg"].values, -w / 2),
                      (m[m.group == g]["mentions_other_neg"].values, w / 2)]:
        ax.scatter(np.full(len(vals), i + off) + jitter(len(vals), .05), 100 * vals,
                   s=14, color="black", alpha=.4, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(["AI virtual\ninfluencers", "Human\ninfluencers"], fontsize=9.5)
ax.set_ylabel("% of comments mentioning\nanother account", fontsize=9.5)
ax.set_title("a   Involvement of other accounts", loc="left", fontsize=11, fontweight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# ---- panel b: artificiality reference, AIVI vs HI ------------------------
ax = axes[1]
a = art[art.n_neg >= MIN_NEG]
for i, (g, col) in enumerate([("AIVI", AIVI_C), ("HI", HI_C)]):
    v = 100 * a[a.group == g].artificiality.values
    ax.scatter(np.full(len(v), i) + jitter(len(v)), v, s=34, color=col,
               edgecolor="white", linewidth=.6, zorder=3)
    ax.hlines(np.median(v), i - .22, i + .22, color="black", linewidth=2, zorder=4)
ax.set_xticks([0, 1])
ax.set_xticklabels(["AI virtual\ninfluencers", "Human\ninfluencers"], fontsize=9.5)
ax.set_ylabel("% of negative comments referring\nto artificiality", fontsize=9.5)
ax.set_title("b   Reference to artificiality", loc="left", fontsize=11, fontweight="bold")
ax.set_xlim(-.5, 1.5)
ax.text(.5, ax.get_ylim()[1] * .94, "δ = 1.00, p < 0.001", ha="center", fontsize=9, color="#333")
ax.spines[["top", "right"]].set_visible(False)

# ---- panel c: within AIVI, negative vs non-negative ---------------------
ax = axes[2]
for _, r in art_w.iterrows():
    ax.plot([0, 1], [100 * r.rate_nonneg, 100 * r.rate_neg], color="#999", linewidth=.9, zorder=2)
ax.scatter(np.zeros(len(art_w)), 100 * art_w.rate_nonneg, s=30, color="#C9C9C9",
           edgecolor="white", linewidth=.6, zorder=3)
ax.scatter(np.ones(len(art_w)), 100 * art_w.rate_neg, s=30, color=AIVI_C,
           edgecolor="white", linewidth=.6, zorder=3)
ax.set_xticks([0, 1])
ax.set_xticklabels(["non-negative\ncomments", "negative\ncomments"], fontsize=9.5)
ax.set_ylabel("% referring to artificiality", fontsize=9.5)
ax.set_title("c   Within AI virtual influencers", loc="left", fontsize=11, fontweight="bold")
ax.set_xlim(-.35, 1.35)
ax.text(.5, ax.get_ylim()[1] * .94, "+4.9 pp, p < 0.001", ha="center", fontsize=9, color="#333")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("output/fig_negative_discourse.png", dpi=600, bbox_inches="tight")
fig.savefig("output/fig_negative_discourse.pdf", bbox_inches="tight")
print("saved output/fig_negative_discourse.{png,pdf}")

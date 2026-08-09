#!/usr/bin/env python3
"""
Assessment of potentially automated commenting activity.

(A) Cohen's kappa for the two-coder manual profile assessment of 280 sampled
    commenters (Data/bot_coding.csv).
(B) Coder-independent behavioral screen: within-commenter self-repetition and
    within-influencer cross-commenter template reuse (identical normalized
    strings of >=4 words posted by >=2 distinct commenters). Reports rates
    per influencer, Mann-Whitney tests and Cliff's delta with bootstrap CIs
    (resampling influencers) comparing the 35 influencer-level rates,
    at-risk-denominator sensitivity versions, leave-one-influencer-out
    stability, and a comment-level clustered logistic regression.

Run from the repository root: python3 automated_activity_screen.py
Reads Data/AIVI_HI.xlsx.zip and Data/bot_coding.csv.
Writes Data/repetition_rates_by_influencer.csv (aggregate rates only, no
raw comment text or commenter usernames).
"""
import io
import re
import unicodedata
import zipfile

import numpy as np
import openpyxl
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

MIN_WORDS = 4
SEED = 12345
N_BOOT = 10000

AIVI_XLSX_ZIP = "Data/AIVI_HI.xlsx.zip"
BOT_CODING_CSV = "Data/bot_coding.csv"
OUT_REPETITION_RATES = "Data/repetition_rates_by_influencer.csv"

TYPE_MAP = {"AI": "AIVI", "HUMAN": "HI"}


# ------------------------------------------------------------------ #
# Shared normalization                                                #
# ------------------------------------------------------------------ #
def normalize(t):
    """Case fold; strip diacritics, punctuation, emoji, and symbols;
    collapse elongated characters (e.g. 'sooooo' -> 'soo')."""
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def cohens_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum()) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    se = np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
    return kappa, po, (kappa - 1.96 * se, kappa + 1.96 * se), m, labels


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))


def hodges_lehmann(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.median((x[:, None] - y[None, :]).ravel())


def bootstrap_ci(x, y, statfunc, n_boot=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        xb = x[rng.randint(0, len(x), len(x))]
        yb = y[rng.randint(0, len(y), len(y))]
        vals[i] = statfunc(xb, yb)
    return np.percentile(vals, [2.5, 97.5])


def section(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------ #
# Data loading                                                         #
# ------------------------------------------------------------------ #
def load_corpus():
    with zipfile.ZipFile(AIVI_XLSX_ZIP) as z:
        xlsx_name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
        wb = openpyxl.load_workbook(io.BytesIO(z.read(xlsx_name)), read_only=True)
    ws = wb["ALL"]
    recs = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        profile, utype, pop, likes, ptime, user, text, clikes, purl, ctext, emoji = row[:11]
        mapped = TYPE_MAP.get(utype)
        if mapped is None:
            continue
        recs.append((profile, mapped, user, text if text is not None else ""))
    df = pd.DataFrame(recs, columns=["influencer", "type", "user", "text"])
    df["_norm"] = df["text"].map(normalize)
    df = df[df["_norm"].str.len() > 0].copy()
    df["_nwords"] = df["_norm"].str.split().str.len()
    return df


def flag_self_repeat(df):
    df = df.copy()
    df["is_self_repeat"] = False
    for grp, idx in df.groupby("type").groups.items():
        sub = df.loc[idx]
        flag = sub.duplicated(subset=["user", "_norm"], keep="first")
        df.loc[idx, "is_self_repeat"] = flag.values
    return df


def flag_template(df):
    df = df.copy()
    df["is_template"] = False
    for infl, idx in df.groupby("influencer").groups.items():
        sub = df.loc[idx]
        longs = sub[sub["_nwords"] >= MIN_WORDS]
        if len(longs) == 0:
            continue
        nuniq = longs.groupby("_norm")["user"].nunique()
        template_norms = set(nuniq[nuniq >= 2].index)
        is_templ = longs["_norm"].isin(template_norms)
        df.loc[longs.index[is_templ.values], "is_template"] = True
    return df


# ------------------------------------------------------------------ #
# (A) Cohen's kappa                                                    #
# ------------------------------------------------------------------ #
def part_a():
    section("(A) INTER-RATER RELIABILITY, MANUAL PROFILE ASSESSMENT")
    try:
        df = pd.read_csv(BOT_CODING_CSV)
    except FileNotFoundError:
        print(f"  [skipped] {BOT_CODING_CSV} not found\n")
        return
    df = df.dropna(subset=["coder_a", "coder_b"])
    k, po, ci, m, labels = cohens_kappa(df["coder_a"], df["coder_b"])
    print(f"  n accounts double-coded : {len(df)}")
    print(f"  percent agreement       : {po*100:.1f}%")
    print(f"  Cohen's kappa           : {k:.3f}  (95% CI {ci[0]:.3f}, {ci[1]:.3f})")
    print(f"  confusion matrix ({labels}):\n{m.astype(int)}")
    bench = ("almost perfect" if k > .80 else "substantial" if k > .60 else
             "moderate" if k > .40 else "fair or lower")
    print(f"  Landis & Koch benchmark : {bench}\n")


# ------------------------------------------------------------------ #
# (B) Behavioral repetition screen                                     #
# ------------------------------------------------------------------ #
def per_influencer_rates(df):
    rows = []
    for infl, sub in df.groupby("influencer"):
        n = len(sub)
        rep = sub["is_self_repeat"].sum()
        longs = sub[sub["_nwords"] >= MIN_WORDS]
        n_long = len(longs)
        templ_excess = 0
        if n_long:
            g = longs.groupby("_norm")["user"].agg(["count", "nunique"])
            templ = g[g["nunique"] >= 2]
            templ_excess = int((templ["count"] - 1).sum())
        rows.append({
            "influencer": infl, "type": sub["type"].iloc[0],
            "comments": n, "self_repeat_%": 100 * rep / n,
            "n_long": n_long,
            "template_cross_%": 100 * templ_excess / n_long if n_long else np.nan,
        })
    return pd.DataFrame(rows)


def group_comparison(name, infl_df, col):
    a = infl_df.loc[infl_df["type"] == "AIVI", col].dropna()
    h = infl_df.loc[infl_df["type"] == "HI", col].dropna()
    u, p = stats.mannwhitneyu(a, h, alternative="two-sided")
    d = cliffs_delta(a, h)
    d_lo, d_hi = bootstrap_ci(a, h, cliffs_delta)
    hl = hodges_lehmann(a, h)
    hl_lo, hl_hi = bootstrap_ci(a, h, hodges_lehmann)
    print(f"  [{name}] AIVI median={a.median():.2f} (IQR {a.quantile(.25):.2f}-{a.quantile(.75):.2f})  "
          f"HI median={h.median():.2f} (IQR {h.quantile(.25):.2f}-{h.quantile(.75):.2f})")
    print(f"    Mann-Whitney U={u:.1f} p={p:.4g}")
    print(f"    Cliff's delta={d:.3f} (bootstrap 95% CI {d_lo:.3f}, {d_hi:.3f}, "
          f"resampling influencers, n_boot={N_BOOT})")
    print(f"    Hodges-Lehmann diff={hl:.3f} pct points (bootstrap 95% CI {hl_lo:.3f}, {hl_hi:.3f})")
    return a, h


def leave_one_out(infl_df, col):
    base_a = infl_df.loc[infl_df["type"] == "AIVI", col].dropna()
    base_h = infl_df.loc[infl_df["type"] == "HI", col].dropna()
    base_u, base_p = stats.mannwhitneyu(base_a, base_h, alternative="two-sided")
    base_d = cliffs_delta(base_a, base_h)
    rows = []
    valid = infl_df.dropna(subset=[col])
    for infl in valid["influencer"]:
        rest = valid[valid["influencer"] != infl]
        a = rest.loc[rest["type"] == "AIVI", col]
        h = rest.loc[rest["type"] == "HI", col]
        u, p = stats.mannwhitneyu(a, h, alternative="two-sided")
        d = cliffs_delta(a, h)
        rows.append({"dropped": infl, "p": p, "delta": d})
    loo = pd.DataFrame(rows)
    n_flip = ((loo["p"] < 0.05) != (base_p < 0.05)).sum()
    print(f"  baseline p={base_p:.4g}, delta={base_d:.3f}")
    print(f"  leave-one-out p range: {loo['p'].min():.4g} - {loo['p'].max():.4g}")
    print(f"  leave-one-out delta range: {loo['delta'].min():.3f} - {loo['delta'].max():.3f}")
    print(f"  influencers whose removal flips significance at 0.05: {n_flip}")


def part_b(df):
    section("(B) AUTOMATED BEHAVIORAL REPETITION SCREEN")
    df = flag_self_repeat(df)
    df = flag_template(df)

    print("\n  --- pooled (comment-level), by group ---")
    grp_rows = []
    for grp, sub in df.groupby("type"):
        n = len(sub)
        rep = sub["is_self_repeat"].sum()
        longs = sub[sub["_nwords"] >= MIN_WORDS]
        g = longs.groupby("_norm")["user"].agg(["count", "nunique"])
        templ = g[g["nunique"] >= 2]
        templ_excess = int((templ["count"] - 1).sum())
        grp_rows.append({"group": grp, "comments": n, "self_repeat_%": 100 * rep / n,
                          "n_long": len(longs), "template_%": 100 * templ_excess / len(longs)})
    gdf = pd.DataFrame(grp_rows).set_index("group")
    print(gdf.round(3).to_string())

    a, h = gdf.loc["AIVI"], gdf.loc["HI"]
    tbl_self = np.array([
        [a["self_repeat_%"] / 100 * a["comments"], a["comments"] - a["self_repeat_%"] / 100 * a["comments"]],
        [h["self_repeat_%"] / 100 * h["comments"], h["comments"] - h["self_repeat_%"] / 100 * h["comments"]],
    ]).round()
    chi2, p, dof, _ = stats.chi2_contingency(tbl_self)
    print(f"\n  self-repeat, comment-level chi2({dof}) = {chi2:.3f}, p = {p:.4g} "
          f"(pseudoreplication contrast; report alongside the influencer-level test)")

    tbl_templ = np.array([
        [a["template_%"] / 100 * a["n_long"], a["n_long"] - a["template_%"] / 100 * a["n_long"]],
        [h["template_%"] / 100 * h["n_long"], h["n_long"] - h["template_%"] / 100 * h["n_long"]],
    ]).round()
    chi2t, pt, doft, _ = stats.chi2_contingency(tbl_templ)
    print(f"  template reuse, comment-level chi2({doft}) = {chi2t:.3f}, p = {pt:.4g}")

    infl_df = per_influencer_rates(df)

    print("\n  --- influencer-level comparison, self-repetition ---")
    group_comparison("self-repeat rate", infl_df, "self_repeat_%")

    print("\n  --- influencer-level comparison, cross-commenter template reuse ---")
    group_comparison("template reuse rate", infl_df, "template_cross_%")

    print("\n  --- at-risk-denominator self-repetition, per influencer ---")
    risk_rows = []
    for infl, sub in df.groupby("influencer"):
        per_user = sub.groupby("user").agg(n=("is_self_repeat", "size"), any_rep=("is_self_repeat", "any"))
        eligible_a = per_user[per_user["n"] >= 2]
        n_elig_a = len(eligible_a)
        rate_a = 100 * eligible_a["any_rep"].mean() if n_elig_a else np.nan
        n_elig_b = int((per_user["n"] - 1).clip(lower=0).sum())
        n_rep_b = int(sub["is_self_repeat"].sum())
        rate_b = 100 * n_rep_b / n_elig_b if n_elig_b else np.nan
        risk_rows.append({"influencer": infl, "type": sub["type"].iloc[0],
                           "n_eligible_a": n_elig_a, "rate_a_%": rate_a,
                           "n_eligible_b": n_elig_b, "rate_b_%": rate_b})
    risk_df = pd.DataFrame(risk_rows)
    risk_df["flag_low_n"] = (risk_df["n_eligible_a"] < 100) | (risk_df["n_eligible_b"] < 100)
    n_flagged = risk_df["flag_low_n"].sum()
    print(f"  flagged (n_eligible < 100 on either version): {n_flagged} influencers")
    for metric in ["rate_a_%", "rate_b_%"]:
        full_a = risk_df.loc[risk_df["type"] == "AIVI", metric].dropna()
        full_h = risk_df.loc[risk_df["type"] == "HI", metric].dropna()
        u1, p1 = stats.mannwhitneyu(full_a, full_h, alternative="two-sided")
        clean = risk_df[~risk_df["flag_low_n"]]
        clean_a = clean.loc[clean["type"] == "AIVI", metric].dropna()
        clean_h = clean.loc[clean["type"] == "HI", metric].dropna()
        u2, p2 = stats.mannwhitneyu(clean_a, clean_h, alternative="two-sided")
        print(f"  {metric}: all 35 -> p={p1:.4g}  |  excluding flagged -> p={p2:.4g} "
              f"(n={len(clean_a)} vs {len(clean_h)})")

    print("\n  --- leave-one-influencer-out, self-repeat rate ---")
    leave_one_out(infl_df, "self_repeat_%")
    print("\n  --- leave-one-influencer-out, template reuse rate ---")
    leave_one_out(infl_df, "template_cross_%")

    print("\n  --- comment-level clustered logistic regression (is_self_repeat) ---")
    per_user_n = df.groupby(["type", "user"]).size().rename("user_n_comments")
    model_df = df.join(per_user_n, on=["type", "user"])
    model_df["log_user_n_comments"] = np.log(model_df["user_n_comments"])
    model_df["char_len"] = model_df["text"].str.len()
    model_df["type_AIVI"] = (model_df["type"] == "AIVI").astype(int)
    model_df["y"] = model_df["is_self_repeat"].astype(int)
    model_df["influencer"] = model_df["influencer"].astype("category")

    m_unadj = smf.logit("y ~ type_AIVI", data=model_df).fit(
        cov_type="cluster", cov_kwds={"groups": model_df["influencer"]}, disp=0)
    m_adj = smf.logit("y ~ type_AIVI + log_user_n_comments + char_len", data=model_df).fit(
        cov_type="cluster", cov_kwds={"groups": model_df["influencer"]}, disp=0)

    for name, res in [("unadjusted", m_unadj), ("adjusted (+log user activity, +char length)", m_adj)]:
        b, se, p = res.params["type_AIVI"], res.bse["type_AIVI"], res.pvalues["type_AIVI"]
        orv = np.exp(b)
        lo, hi = np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
        print(f"  [{name}] type_AIVI OR={orv:.3f} (95% CI {lo:.3f}-{hi:.3f}) p={p:.4g}, "
              f"clustered by influencer (n_clusters={model_df['influencer'].nunique()})")

    infl_df.round(3).to_csv(OUT_REPETITION_RATES, index=False)
    print(f"\n  saved per-influencer rates to {OUT_REPETITION_RATES}")


if __name__ == "__main__":
    part_a()
    df = load_corpus()
    part_b(df)

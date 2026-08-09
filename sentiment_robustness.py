#!/usr/bin/env python3
"""
Sensitivity analysis of the categorical sentiment comparisons across four
corpora: the original analysis corpus, a corpus excluding comments with no
text after normalization, a corpus excluding comments flagged by either
repetition criterion, and a corpus excluding both.

Reuses the stored per-comment discrete sentiment label produced by
data_analysis.R (Data/sentiment_analysis_results1.csv.zip) -- no scores are
recomputed. Reports prevalence, percentage-point differences, Cohen's h and
odds ratios for each condition, plus the polarity comparison on the stored
sentiment variable and Bonferroni-corrected category tests.

Run from the repository root: python3 sentiment_robustness.py
Reads Data/sentiment_analysis_results1.csv.zip.
Writes Data/sentiment_robustness_summary.csv (aggregate rates only, no raw
comment text or commenter usernames).
"""
import io
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

MIN_WORDS = 4
SENT_CSV_ZIP = "Data/sentiment_analysis_results1.csv.zip"
OUT_SUMMARY = "Data/sentiment_robustness_summary.csv"

TYPE_MAP = {"AI": "AIVI", "HUMAN": "HI"}


def normalize(t):
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def odds_ratio(a_yes, a_no, h_yes, h_no):
    return (a_yes / a_no) / (h_yes / h_no)


def bonferroni(pvals, m):
    return [min(p * m, 1.0) for p in pvals]


def section(title):
    print("=" * 74)
    print(title)
    print("=" * 74)


def load():
    with zipfile.ZipFile(SENT_CSV_ZIP) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        raw = z.read(csv_name)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding="utf-8", encoding_errors="replace")
    df = df[["PROFILE.url", "user_type", "posts.comments.user", "posts.comments.text", "sentiment"]].copy()
    df.columns = ["influencer", "type_raw", "user", "text", "sentiment"]
    df["type"] = df["type_raw"].map(TYPE_MAP)
    df["_norm"] = df["text"].map(normalize)
    df["_nwords"] = df["_norm"].str.split().str.len()
    df["_empty"] = df["_norm"].str.len() == 0
    return df


def flag_repetition(df):
    df = df.copy()
    df["is_self_repeat"] = False
    df["is_template"] = False
    scorable = df[~df["_empty"]]
    for grp, idx in scorable.groupby("type").groups.items():
        sub = df.loc[idx]
        flag = sub.duplicated(subset=["user", "_norm"], keep="first")
        df.loc[idx, "is_self_repeat"] = flag.values
    for infl, idx in scorable.groupby("influencer").groups.items():
        sub = df.loc[idx]
        longs = sub[sub["_nwords"] >= MIN_WORDS]
        if len(longs) == 0:
            continue
        nuniq = longs.groupby("_norm")["user"].nunique()
        template_norms = set(nuniq[nuniq >= 2].index)
        is_templ = longs["_norm"].isin(template_norms)
        df.loc[longs.index[is_templ.values], "is_template"] = True
    return df


def cat_props(sub):
    n = len(sub)
    neg = (sub["sentiment"] == -1).sum()
    neu = (sub["sentiment"] == 0).sum()
    pos = (sub["sentiment"] == 1).sum()
    return n, neg, neu, pos


def main():
    df = load()
    print(f"loaded: {len(df)} rows, {df['influencer'].nunique()} influencers")
    df = flag_repetition(df)

    df["flag_nontext"] = df["_empty"]
    df["flag_repetition"] = df["is_self_repeat"] | df["is_template"]
    df["flag_conservative"] = df["flag_nontext"] | df["flag_repetition"]

    corpora = {
        "ORIGINAL": df,
        "TEXT-SCORABLE": df[~df["flag_nontext"]],
        "REPETITION-FILTERED": df[~df["flag_repetition"]],
        "CONSERVATIVE": df[~df["flag_conservative"]],
    }

    section("FOUR-CORPORA SENTIMENT ROBUSTNESS")
    summary_rows = []
    for name, corpus in corpora.items():
        a = corpus[corpus["type"] == "AIVI"]
        h = corpus[corpus["type"] == "HI"]
        na, nega, neua, posa = cat_props(a)
        nh, negh, neuh, posh = cat_props(h)

        w, p_w = stats.mannwhitneyu(a["sentiment"], h["sentiment"], alternative="two-sided")
        tbl2x3 = np.array([[nega, neua, posa], [negh, neuh, posh]])
        chi2_omni, p_omni, _, _ = stats.chi2_contingency(tbl2x3)

        pvals, stats_cat = [], {}
        for cat, (ca, ch) in [("negative", (nega, negh)), ("neutral", (neua, neuh)), ("positive", (posa, posh))]:
            tbl = np.array([[ca, na - ca], [ch, nh - ch]])
            chi2c, pc, _, _ = stats.chi2_contingency(tbl)
            pvals.append(pc)
            stats_cat[cat] = pc
        p_adj = bonferroni(pvals, 3)

        h_neutral = cohens_h(neua / na, neuh / nh)
        or_neutral = odds_ratio(neua, na - neua, neuh, nh - neuh)
        h_positive = cohens_h(posa / na, posh / nh)
        or_positive = odds_ratio(posa, na - posa, posh, nh - posh)

        print(f"\n  --- {name} ---  n(AIVI)={na}  n(HI)={nh}")
        print(f"  neutral%: AIVI={100*neua/na:.2f}  HI={100*neuh/nh:.2f}  "
              f"diff={100*neua/na - 100*neuh/nh:+.2f}pp  h={h_neutral:.3f}  OR={or_neutral:.3f}")
        print(f"  positive%: AIVI={100*posa/na:.2f}  HI={100*posh/nh:.2f}  "
              f"diff={100*posa/na - 100*posh/nh:+.2f}pp  h={h_positive:.3f}  OR={or_positive:.3f}")
        print(f"  negative%: AIVI={100*nega/na:.2f}  HI={100*negh/nh:.2f}")
        print(f"  polarity (discretized sentiment) Mann-Whitney: W={w:.4e}  p={p_w:.4e}")
        print(f"  omnibus 2x3 chi2 = {chi2_omni:.2f}  p={p_omni:.4e}")
        print(f"  category tests, Bonferroni-adjusted (m=3): "
              f"neg={p_adj[0]:.4e}  neu={p_adj[1]:.4e}  pos={p_adj[2]:.4e}")

        summary_rows.append({
            "corpus": name, "n_AIVI": na, "n_HI": nh,
            "neutral_AIVI_%": 100*neua/na, "neutral_HI_%": 100*neuh/nh,
            "positive_AIVI_%": 100*posa/na, "positive_HI_%": 100*posh/nh,
            "negative_AIVI_%": 100*nega/na, "negative_HI_%": 100*negh/nh,
            "cohens_h_neutral": h_neutral, "OR_neutral": or_neutral,
            "cohens_h_positive": h_positive, "OR_positive": or_positive,
            "polarity_MW_p": p_w, "p_neutral_adj": p_adj[1], "p_positive_adj": p_adj[2],
        })

    summary_df = pd.DataFrame(summary_rows).set_index("corpus")
    print("\n  === SUMMARY TABLE ACROSS CORPORA ===")
    print(summary_df.round(4).to_string())
    summary_df.round(4).to_csv(OUT_SUMMARY)
    print(f"\n  saved summary table to {OUT_SUMMARY}")


if __name__ == "__main__":
    main()

"""
@mention indicators for comments classified as negative.

For each comment, four binary indicators are derived: any @mention, a mention of
the influencer's own handle, a mention of any other account, and an @mention in
initial position. Rates are compared within influencer (negative vs non-negative
comments) and between influencer types.

Usage:  python mention_analysis.py --data <sentiment_analysis_results1.csv>
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_BOOT = 10000
MIN_NEG = 30            # minimum negative comments per influencer for inclusion
INDICATORS = ["any_mention", "mentions_other", "mention_initial", "mentions_self"]

MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z0-9._]{1,30})")
# leading characters that may precede an initial mention (whitespace, emoji, quotes)
LEADING_JUNK_RE = re.compile(r"^[\s\"'​-‍️\U0001F000-\U0001FAFF☀-➿]+")


def extract_mentions(text):
    """Return list of distinct mentioned handles, trailing periods stripped."""
    if not isinstance(text, str):
        return []
    out, seen = [], set()
    for m in MENTION_RE.findall(text):
        h = m.rstrip(".").lower()
        if h and h not in seen:
            seen.add(h)
            out.append(h)
    return out


def starts_with_mention(text):
    """True if the first textual element of the comment is an @mention."""
    if not isinstance(text, str):
        return False
    return LEADING_JUNK_RE.sub("", text).startswith("@")


def normalize_handle(h):
    return h.replace(".", "").replace("_", "").lower()


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = np.sign(a[:, None] - b[None, :])
    return diff.mean()


def boot_ci_two_group(a, b, stat_fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    vals = [stat_fn(rng.choice(a, len(a), replace=True),
                    rng.choice(b, len(b), replace=True)) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_ci_one_group(x, stat_fn=np.median, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    vals = [stat_fn(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rank_biserial_paired(d):
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank test."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    rp, rn = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((rp - rn) / ranks.sum())


def load(path):
    df = pd.read_csv(path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df["handle"] = (df["PROFILE.url"].astype(str).str.rstrip("/")
                    .str.split("/").str[-1].str.lower())
    df["text"] = df["posts.comments.text"].fillna("").astype(str)
    df["group"] = df["user_type"].map({"AI": "AIVI", "HUMAN": "HI"})

    # verification against Table 3 of the manuscript
    counts = df[df.sentiment == -1].groupby("group").size().to_dict()
    expected = {"AIVI": 4396, "HI": 7486}
    if counts != expected:
        raise SystemExit(f"Negative counts {counts} do not reproduce Table 3 {expected}. Stopping.")
    print(f"[verify] negative comments reproduce Table 3: {counts}")
    return df


def build_indicators(df):
    df = df.copy()
    df["mention_list"] = df["text"].map(extract_mentions)
    df["n_mentions"] = df["mention_list"].map(len)
    df["any_mention"] = df["n_mentions"] > 0

    # primary self-mention definition: exact case-insensitive handle match
    df["mentions_self"] = [any(m == h for m in ms)
                           for ms, h in zip(df["mention_list"], df["handle"])]
    df["mentions_other"] = [any(m != h for m in ms)
                            for ms, h in zip(df["mention_list"], df["handle"])]

    # sensitivity: normalized matching (periods/underscores removed)
    df["mentions_self_norm"] = [any(normalize_handle(m) == normalize_handle(h) for m in ms)
                                for ms, h in zip(df["mention_list"], df["handle"])]
    df["mentions_other_norm"] = [
        any(normalize_handle(m) != normalize_handle(h) for m in ms)
        for ms, h in zip(df["mention_list"], df["handle"])]

    df["mention_initial"] = df["text"].map(starts_with_mention)
    df["is_neg"] = df["sentiment"] == -1
    return df


def influencer_rates(df, indicators):
    """Per-influencer rates among negative, non-negative and all comments."""
    rows = []
    for (grp, h), g in df.groupby(["group", "handle"]):
        neg, non = g[g.is_neg], g[~g.is_neg]
        r = {"group": grp, "handle": h, "n_neg": len(neg), "n_nonneg": len(non), "n_all": len(g)}
        for ind in indicators:
            r[f"{ind}_neg"] = neg[ind].mean() if len(neg) else np.nan
            r[f"{ind}_nonneg"] = non[ind].mean() if len(non) else np.nan
            r[f"{ind}_all"] = g[ind].mean()
            r[f"{ind}_diff"] = r[f"{ind}_neg"] - r[f"{ind}_nonneg"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["group", "handle"]).reset_index(drop=True)


def run_tests(rates, indicators, restricted, n_ind):
    """Within-AIVI test and AIVI-vs-HI comparison."""
    sub = rates[rates.n_neg >= MIN_NEG] if restricted else rates
    a = sub[sub.group == "AIVI"]
    h = sub[sub.group == "HI"]
    out = []
    for ind in indicators:
        da = a[f"{ind}_diff"].dropna().values
        dh = h[f"{ind}_diff"].dropna().values

        # within AIVI: does P(mention|neg) - P(mention|non-neg) differ from 0?
        if len(da) >= 3 and np.any(da != 0):
            w_stat, w_p = stats.wilcoxon(da, zero_method="wilcox", alternative="two-sided")
        else:
            w_stat, w_p = np.nan, np.nan
        lo_a, hi_a = boot_ci_one_group(da) if len(da) else (np.nan, np.nan)

        # between groups: AIVI vs HI within-influencer differences
        if len(da) and len(dh):
            u_stat, u_p = stats.mannwhitneyu(da, dh, alternative="two-sided")
            delta = cliffs_delta(da, dh)
            lo_d, hi_d = boot_ci_two_group(da, dh, cliffs_delta)
        else:
            u_stat = u_p = delta = lo_d = hi_d = np.nan

        out.append({
            "indicator": ind,
            "analysis": "restricted (n_neg>=30)" if restricted else "all influencers",
            "n_AIVI": len(da), "n_HI": len(dh),
            "AIVI_rate_neg": a[f"{ind}_neg"].median(),
            "AIVI_rate_nonneg": a[f"{ind}_nonneg"].median(),
            "AIVI_median_diff": float(np.median(da)) if len(da) else np.nan,
            "AIVI_diff_CI_low": lo_a, "AIVI_diff_CI_high": hi_a,
            "AIVI_wilcoxon_p": w_p,
            "AIVI_wilcoxon_p_bonf": min(w_p * n_ind, 1.0) if w_p == w_p else np.nan,
            "AIVI_rank_biserial": rank_biserial_paired(da),
            "HI_rate_neg": h[f"{ind}_neg"].median(),
            "HI_rate_nonneg": h[f"{ind}_nonneg"].median(),
            "HI_median_diff": float(np.median(dh)) if len(dh) else np.nan,
            "AIVIvsHI_mannwhitney_p": u_p,
            "AIVIvsHI_mannwhitney_p_bonf": min(u_p * n_ind, 1.0) if u_p == u_p else np.nan,
            "AIVIvsHI_cliffs_delta": delta,
            "AIVIvsHI_delta_CI_low": lo_d, "AIVIvsHI_delta_CI_high": hi_d,
        })
    return pd.DataFrame(out)


def top_mentioned(df, out_dir, k=50):
    rows = []
    for grp, g in df[df.is_neg].groupby("group"):
        c = {}
        for ms, h in zip(g["mention_list"], g["handle"]):
            for m in ms:
                if m != h:
                    c[m] = c.get(m, 0) + 1
        for handle, n in sorted(c.items(), key=lambda x: -x[1])[:k]:
            rows.append({"group": grp, "mentioned_handle": handle, "n_negative_comments": n})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "top_mentioned_handles.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = build_indicators(load(args.data))

    rates = influencer_rates(df, INDICATORS + ["mentions_other_norm", "mentions_self_norm"])
    rates.to_csv(os.path.join(args.out, "mention_analysis_by_influencer.csv"), index=False)

    res = pd.concat([
        run_tests(rates, INDICATORS, restricted=True, n_ind=len(INDICATORS)),
        run_tests(rates, INDICATORS, restricted=False, n_ind=len(INDICATORS)),
        run_tests(rates, ["mentions_other_norm", "mentions_self_norm"],
                  restricted=True, n_ind=len(INDICATORS)).assign(
                      analysis="sensitivity: normalized handle matching"),
    ], ignore_index=True)
    res.to_csv(os.path.join(args.out, "mention_analysis_summary.csv"), index=False)

    # comment-level descriptives
    desc = (df[df.is_neg].groupby("group")[INDICATORS].mean() * 100).round(2)
    desc_all = (df.groupby("group")[INDICATORS].mean() * 100).round(2)
    desc.to_csv(os.path.join(args.out, "mention_comment_level_negatives.csv"))
    desc_all.to_csv(os.path.join(args.out, "mention_comment_level_allcomments.csv"))

    top_mentioned(df, args.out)

    excl = rates[rates.n_neg < MIN_NEG].groupby("group").handle.apply(list).to_dict()
    print("\n[comment-level %, negatives]\n", desc)
    print("\n[comment-level %, all comments (descriptive baseline)]\n", desc_all)
    print("\n[excluded by n_neg<30]", excl)
    print("\n[tests]\n", res[["indicator", "analysis", "AIVI_median_diff", "AIVI_wilcoxon_p_bonf",
                              "AIVIvsHI_cliffs_delta", "AIVIvsHI_mannwhitney_p_bonf"]].to_string())
    df.to_pickle(os.path.join(args.out, "_indicators.pkl"))


if __name__ == "__main__":
    main()

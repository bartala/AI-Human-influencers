"""
Reference-to-artificiality analysis of comments classified as negative.

A pre-specified term list denoting artificial status (ai, robot, bot, cgi,
avatar, virtual, artificial, computer-generated, deepfake, android, cyborg, and
explicit negations of realness or humanness) is matched against raw comment text
using case-insensitive regular expressions with word boundaries.

Two tests, both at the influencer level:
  (1) Between groups: AIVI vs HI reference rate among negative comments.
  (2) Within AIVI: P(reference | negative) - P(reference | non-negative).

Usage:  python artificiality_keywords.py --data <sentiment_analysis_results1.csv>
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_BOOT = 10000
MIN_NEG = 30          # minimum negative comments per influencer for inclusion

ARTIFICIALITY = (
    r"\b(ai|a\.i\.|robot|robots|bot|bots|cgi|avatar|virtual|artificial|"
    r"computer generated|computer-generated|ai generated|ai-generated|"
    r"deepfake|deep fake|android|cyborg|not (a )?real|isn'?t real|"
    r"is she real|is it real|not human|fake person)\b"
)

TERM_BREAKDOWN = {
    "ai": r"\bai\b",
    "a.i.": r"\ba\.i\.",
    "robot(s)": r"\brobots?\b",
    "bot(s)": r"\bbots?\b",
    "cgi": r"\bcgi\b",
    "avatar": r"\bavatar\b",
    "virtual": r"\bvirtual\b",
    "artificial": r"\bartificial\b",
    "computer-/AI-generated": r"\b(computer|ai)[- ]generated\b",
    "deepfake": r"\bdeep ?fake\b",
    "android/cyborg": r"\b(android|cyborg)\b",
    "not real / not human / is she real": r"\b(not (a )?real|isn'?t real|is she real|"
                                          r"is it real|not human|fake person)\b",
}


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return np.sign(a[:, None] - b[None, :]).mean()


def boot_ci_two(a, b, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = [cliffs_delta(rng.choice(a, len(a), True), rng.choice(b, len(b), True)) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_ci_one(x, fn=np.median, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    v = [fn(rng.choice(x, len(x), True)) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def between_groups(neg, label, min_neg=MIN_NEG):
    """Test (1): AIVI vs HI reference rate among negative comments."""
    g = (neg.groupby(["group", "handle"])
         .agg(n_neg=("artificiality", "size"), rate=("artificiality", "mean")).reset_index())
    g = g[g.n_neg >= min_neg]
    x = g[g.group == "AIVI"].rate.values
    y = g[g.group == "HI"].rate.values
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    lo, hi = boot_ci_two(x, y)
    return {"analysis": label, "n_AIVI": len(x), "n_HI": len(y),
            "AIVI_median_pct": 100 * np.median(x), "HI_median_pct": 100 * np.median(y),
            "mannwhitney_U": u, "mannwhitney_p": p,
            "cliffs_delta": cliffs_delta(x, y), "CI_low": lo, "CI_high": hi}


def within_aivi(df, min_neg=MIN_NEG):
    """Test (2): within AIVI, P(ref|negative) - P(ref|non-negative)."""
    rows = []
    for h, sub in df[df.group == "AIVI"].groupby("handle"):
        neg, non = sub[sub.is_neg], sub[~sub.is_neg]
        if len(neg) < min_neg or len(non) == 0:
            continue
        rows.append({"handle": h, "n_neg": len(neg),
                     "rate_neg": neg.artificiality.mean(),
                     "rate_nonneg": non.artificiality.mean(),
                     "diff": neg.artificiality.mean() - non.artificiality.mean()})
    r = pd.DataFrame(rows)
    w, p = stats.wilcoxon(r["diff"].values, zero_method="wilcox", alternative="two-sided")
    lo, hi = boot_ci_one(r["diff"].values)
    return r, {"n_influencers": len(r),
               "median_rate_neg_pct": 100 * r.rate_neg.median(),
               "median_rate_nonneg_pct": 100 * r.rate_nonneg.median(),
               "median_diff_pp": 100 * r["diff"].median(),
               "diff_CI_low_pp": 100 * lo, "diff_CI_high_pp": 100 * hi,
               "wilcoxon_W": w, "wilcoxon_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.data, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df["handle"] = df["PROFILE.url"].astype(str).str.rstrip("/").str.split("/").str[-1].str.lower()
    df["group"] = df["user_type"].map({"AI": "AIVI", "HUMAN": "HI"})
    df["text"] = df["posts.comments.text"].fillna("").astype(str)
    df["is_neg"] = df["sentiment"] == -1
    df["artificiality"] = df["text"].str.lower().str.contains(ARTIFICIALITY, regex=True, na=False)

    neg = df[df.is_neg].copy()
    counts = neg.groupby("group").size().to_dict()
    if counts != {"AIVI": 4396, "HI": 7486}:
        raise SystemExit(f"Negative counts {counts} do not reproduce Table 3. Stopping.")
    print(f"[verify] negatives reproduce Table 3: {counts}")

    # ---- comment-level descriptives -------------------------------------
    desc = pd.DataFrame({
        "negative_pct": 100 * neg.groupby("group").artificiality.mean(),
        "non_negative_pct": 100 * df[~df.is_neg].groupby("group").artificiality.mean(),
        "all_comments_pct": 100 * df.groupby("group").artificiality.mean(),
    }).round(2)
    desc.to_csv(os.path.join(a.out, "artificiality_comment_level.csv"))
    print("\n[comment-level %] comments containing artificiality terms\n", desc.to_string())

    # ---- influencer-level rates -----------------------------------------
    infl = (neg.groupby(["group", "handle"])
            .agg(n_neg=("artificiality", "size"), artificiality=("artificiality", "mean"))
            .reset_index())
    infl.to_csv(os.path.join(a.out, "artificiality_by_influencer.csv"), index=False)
    excluded = infl[infl.n_neg < MIN_NEG].groupby("group").handle.apply(list).to_dict()
    print(f"\n[excluded by n_neg<{MIN_NEG}] {excluded}")

    # ---- test 1: between groups ------------------------------------------
    res = [between_groups(neg, f"primary (n_neg>={MIN_NEG})"),
           between_groups(neg, "sensitivity: all 35 influencers", min_neg=0)]

    # sensitivity: English-detected comments only
    try:
        import py3langid as langid
        neg = neg.copy()
        neg["lang"] = [langid.classify(t)[0] if len(t) >= 3 else "und"
                       for t in neg["text"].tolist()]
        res.append(between_groups(neg[neg.lang == "en"], "sensitivity: English-detected only"))
    except ImportError:
        print("[warn] py3langid not installed; skipping language sensitivity check")

    res = pd.DataFrame(res)
    res.to_csv(os.path.join(a.out, "artificiality_tests_between.csv"), index=False)
    print("\n[test 1: AIVI vs HI, negative comments]\n", res.round(4).to_string(index=False))

    # ---- test 2: within AIVI, negative vs non-negative -------------------
    per_infl, w = within_aivi(df)
    per_infl.to_csv(os.path.join(a.out, "artificiality_within_aivi.csv"), index=False)
    pd.DataFrame([w]).to_csv(os.path.join(a.out, "artificiality_tests_within.csv"), index=False)
    print("\n[test 2: within AIVI, negative vs non-negative]")
    for k, v in w.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- term breakdown ---------------------------------------------------
    br = []
    low = neg["text"].str.lower()
    for t, pat in TERM_BREAKDOWN.items():
        m = low.str.contains(pat, regex=True, na=False)
        br.append({"term": t,
                   "AIVI_n": int((m & (neg.group == "AIVI")).sum()),
                   "HI_n": int((m & (neg.group == "HI")).sum())})
    br = pd.DataFrame(br).sort_values("AIVI_n", ascending=False)
    br.to_csv(os.path.join(a.out, "artificiality_term_breakdown.csv"), index=False)
    print("\n[term breakdown, negative comments]\n", br.to_string(index=False))



if __name__ == "__main__":
    main()

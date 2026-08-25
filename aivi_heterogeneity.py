"""
Heterogeneity in affective engagement across influencers.

Primary outcome: whether a comment carries affective expression (positive or
negative polarity). Secondary: positive, negative and neutral rates.

Sampling frame: the 100 most recent posts per influencer, as collected,
restricted to English-language comments in line with the sentiment lexicon.

Analyses:
  (1) Does affective engagement differ across AI virtual influencers (AIVIs)?
      GEE logistic regression with comments clustered within posts and the
      influencer as a categorical predictor; Kruskal-Wallis on post-level
      rates as a distribution-free check.
  (2) Is the dispersion of influencer-level rates across AIVIs different from
      the dispersion across human influencers (HIs)?
  (3) Which observable characteristics covary with affective engagement? Eight
      are derived from the same posts, including the salience of artificiality
      in an influencer's comment stream (term list of artificiality_keywords.py);
      the ninth is Instagram account age, taken from the platform's "Date joined"
      field and expressed in whole months to November 2025.

Usage:  python aivi_heterogeneity.py --data <sentiment_analysis_results1.csv>
"""

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SEED = 42
N_BOOT = 10000
MIN_ENGLISH = 100          # minimum English comments for a stable influencer rate
MIN_TOKENS = 5             # minimum tokens for reliable language detection
REF_DATE = (2025, 11)      # reference point for account age (year, month)

# Instagram "Date joined", month and year, as displayed on each profile.
JOINED = {
    "fit_aitana": (2023, 6), "alizarexx": (2019, 3), "amara_gram": (2017, 4),
    "bermudaisbae": (2016, 8), "dagny.gram": (2019, 2), "itsellastoller": (2019, 10),
    "ezmusgita": (2015, 11), "imma.gram": (2018, 7), "ivaany.h": (2013, 3),
    "opalslutuniverse": (2018, 4), "kda_music": (2020, 7), "lilaziyagil": (2015, 11),
    "magazineluiza": (2014, 4), "millasofiafin": (2022, 11), "lilmiquela": (2016, 4),
    "naina_avtr": (2022, 9), "ria_ria_tokyo": (2019, 2), "sasha_____she": (2020, 1),
    "iamsatiko_": (2021, 8), "shudu.gram": (2016, 11), "sofielund98": (2023, 12),
    "thalasya_": (2018, 11), "zoedvir": (2018, 10),
    "growingannanas": (2012, 8), "darcymcqueeny": (2012, 8), "haleyybaylee": (2012, 7),
    "jessvalortiz": (2012, 5), "minnahigh": (2013, 10), "madeleinecwhite": (2011, 5),
    "meicrosoft": (2015, 6), "mia_irl": (2018, 11), "iamnatalie": (2012, 12),
    "nikkiglamour": (2013, 1), "shalicenoel": (2012, 1), "veronikarajek": (2014, 4),
}


def account_age_months(handle):
    y, m = JOINED[handle]
    return (REF_DATE[0] - y) * 12 + (REF_DATE[1] - m)


MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9._]{1,30}")

ARTIFICIALITY = (
    r"\b(ai|a\.i\.|robot|robots|bot|bots|cgi|avatar|virtual|artificial|"
    r"computer generated|computer-generated|ai generated|ai-generated|"
    r"deepfake|deep fake|android|cyborg|not (a )?real|isn'?t real|"
    r"is she real|is it real|not human|fake person)\b"
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(path):
    df = pd.read_csv(path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df["handle"] = (df["PROFILE.url"].astype(str).str.rstrip("/")
                    .str.split("/").str[-1].str.lower())
    df["group"] = df["user_type"].map({"AI": "AIVI", "HUMAN": "HI"})
    df["text"] = df["posts.comments.text"].fillna("").astype(str)
    df["post"] = df["posts.post_url"].astype(str)
    df["affect"] = (df["sentiment"] != 0).astype(int)
    df["pos"] = (df["sentiment"] == 1).astype(int)
    df["neg"] = (df["sentiment"] == -1).astype(int)
    df["neu"] = (df["sentiment"] == 0).astype(int)
    df["artificiality"] = df["text"].str.lower().str.contains(
        ARTIFICIALITY, regex=True, na=False).astype(int)
    counts = df[df.sentiment == -1].groupby("group").size().to_dict()
    if counts != {"AIVI": 4396, "HI": 7486}:
        raise SystemExit(f"Negative counts {counts} do not reproduce Table 3. Stopping.")
    print(f"[verify] negative counts reproduce Table 3: {counts}")
    return df


def english_frame(df):
    """English-language comments long enough for reliable detection."""
    import py3langid as langid
    clean = df["text"].map(lambda t: re.sub(r"\s+", " ", MENTION_RE.sub(" ", t)).strip())
    ntok = clean.str.split().map(len)
    lang = np.where(ntok >= MIN_TOKENS,
                    [langid.classify(t)[0] if len(t.split()) >= MIN_TOKENS else "und"
                     for t in clean], "und")
    d = df[(ntok >= MIN_TOKENS) & (lang == "en")].copy()
    keep = d.groupby("handle").size()
    return d[d.handle.isin(keep[keep >= MIN_ENGLISH].index)].copy()


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #
def cluster_bootstrap_ci(sub, col, n=N_BOOT, seed=SEED):
    """Percentile interval from resampling posts, the clustering unit."""
    rng = np.random.default_rng(seed)
    posts = sub["post"].unique()
    by_post = {p: g[col].values for p, g in sub.groupby("post")}
    vals = [np.concatenate([by_post[p] for p in rng.choice(posts, len(posts), True)]).mean()
            for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def descriptives(df):
    rows = []
    for (g, h), sub in df.groupby(["group", "handle"]):
        lo, hi = cluster_bootstrap_ci(sub, "affect")
        rows.append({
            "group": g, "handle": h,
            "n_posts": sub["post"].nunique(), "n_comments": len(sub),
            "affective_rate": sub.affect.mean(), "ci_low": lo, "ci_high": hi,
            "positive_rate": sub.pos.mean(), "negative_rate": sub.neg.mean(),
            "neutral_rate": sub.neu.mean(),
        })
    return (pd.DataFrame(rows).sort_values("affective_rate", ascending=False)
            .reset_index(drop=True))


def gee_test(d, outcome="affect"):
    """GEE logistic, influencer as predictor, comments clustered within posts."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    x = d[[outcome, "handle", "post"]].copy()
    x["post_id"] = pd.factorize(x["post"])[0]
    x = x.sort_values("post_id")
    r = smf.gee(f"{outcome} ~ C(handle)", groups="post_id", data=x,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    names = [n for n in r.params.index if "handle" in n]
    R = np.zeros((len(names), len(r.params)))
    for i, n in enumerate(names):
        R[i, list(r.params.index).index(n)] = 1
    w = r.wald_test(R, scalar=False)
    return {"statistic": float(np.squeeze(w.statistic)), "df": len(names),
            "p": float(np.squeeze(w.pvalue)),
            "n_comments": len(x), "n_posts": x.post_id.nunique()}


def kruskal_post_level(d, outcome="affect"):
    post = d.groupby(["handle", "post"])[outcome].mean().reset_index()
    groups = [g[outcome].values for _, g in post.groupby("handle")]
    H, p = stats.kruskal(*groups)
    k, N = len(groups), len(post)
    return {"H": float(H), "df": k - 1, "p": float(p),
            "epsilon_squared": float((H - k + 1) / (N - k)), "n_posts": N}, post


def spread_comparison(desc, n=N_BOOT, seed=SEED):
    """Dispersion of influencer-level rates, AIVI vs HI, with a bootstrap CI."""
    rng = np.random.default_rng(seed)
    a = desc.loc[desc.group == "AIVI", "affective_rate"].values
    h = desc.loc[desc.group == "HI", "affective_rate"].values
    diffs = [rng.choice(a, len(a), True).std(ddof=1) - rng.choice(h, len(h), True).std(ddof=1)
             for _ in range(n)]
    fl_H, fl_p = stats.fligner(a, h)
    return {
        "n_AIVI": len(a), "n_HI": len(h),
        "AIVI_sd": a.std(ddof=1), "HI_sd": h.std(ddof=1),
        "sd_difference": a.std(ddof=1) - h.std(ddof=1),
        "sd_diff_CI_low": float(np.percentile(diffs, 2.5)),
        "sd_diff_CI_high": float(np.percentile(diffs, 97.5)),
        "AIVI_range_low": a.min(), "AIVI_range_high": a.max(),
        "HI_range_low": h.min(), "HI_range_high": h.max(),
        "AIVI_IQR": np.subtract(*np.percentile(a, [75, 25])),
        "HI_IQR": np.subtract(*np.percentile(h, [75, 25])),
        "fligner_chi2": float(fl_H), "fligner_p": float(fl_p),
    }


def boot_spearman_ci(x, y, n=N_BOOT, seed=SEED):
    """Percentile interval for Spearman's rho, resampling influencers."""
    rng = np.random.default_rng(seed)
    x, y, k, v = np.asarray(x, float), np.asarray(y, float), len(x), []
    for _ in range(n):
        i = rng.choice(k, k, True)
        if len(set(x[i])) < 3 or len(set(y[i])) < 3:
            continue
        v.append(stats.spearmanr(x[i], y[i]).statistic)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def leave_one_out_spearman(x, y):
    """Spearman's rho recomputed with each influencer removed in turn."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return [stats.spearmanr(np.delete(x, i), np.delete(y, i)).statistic
            for i in range(len(x))]


def age_correlations(m):
    """Account age against positive and overall affective expression."""
    rows = []
    for g in ["AIVI", "HI"]:
        s = m[m.group == g]
        for outcome in ["positive_rate", "affective_rate"]:
            x, y = s.account_age_months.values, s[outcome].values
            r = stats.spearmanr(x, y)
            lo, hi = boot_spearman_ci(x, y)
            l = leave_one_out_spearman(x, y)
            rows.append({"group": g, "outcome": outcome, "n": len(s),
                         "spearman_rho": r.statistic, "p": r.pvalue,
                         "CI_low": lo, "CI_high": hi,
                         "loo_min": min(l), "loo_max": max(l)})
    return pd.DataFrame(rows)


def characteristics(df, eng):
    """Observable influencer characteristics."""
    rows = []
    for (g, h), sub in eng.groupby(["group", "handle"]):
        posts = sub.drop_duplicates("post")
        t = pd.to_datetime(df.loc[df.handle == h, "posts.time"],
                           format="%d/%m/%Y", errors="coerce").dropna()
        span = (t.max() - t.min()).days if len(t) > 1 else np.nan
        rows.append({
            "group": g, "handle": h,
            "followers": pd.to_numeric(sub["popularity"], errors="coerce").iloc[0],
            "mean_likes_per_post": pd.to_numeric(
                posts["posts.likes_count"].astype(str).str.replace(",", ""),
                errors="coerce").mean(),
            "mean_likes_per_comment": pd.to_numeric(
                sub["pot.comment.likes_count"], errors="coerce").mean(),
            "comments_per_post": len(sub) / sub["post"].nunique(),
            "mean_comment_length": sub["text"].str.len().mean(),
            "distinct_commenter_share": sub["posts.comments.user"].nunique() / len(sub),
            "posts_per_month": (30 * t.nunique() / span) if span and span > 0 else np.nan,
            "artificiality_salience": sub.artificiality.mean(),
            "account_age_months": account_age_months(h),
        })
    return pd.DataFrame(rows)


def correlate(m, cols, group=None, label=""):
    d = m if group is None else m[m.group == group]
    out = []
    for c in cols:
        s = d.dropna(subset=[c, "affective_rate"])
        rho, p = stats.spearmanr(s[c], s.affective_rate)
        out.append({"subset": label, "characteristic": c, "n": len(s),
                    "spearman_rho": rho, "p": p,
                    "p_bonferroni": min(p * len(cols), 1.0)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output_het")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.data)
    eng = english_frame(df)
    print(f"[frame] English comments: {eng.handle.nunique()} influencers "
          f"({eng[eng.group=='AIVI'].handle.nunique()} AIVI, "
          f"{eng[eng.group=='HI'].handle.nunique()} HI), "
          f"{eng['post'].nunique()} posts, {len(eng)} comments")

    # ---- ranking ---------------------------------------------------------
    desc = descriptives(eng)
    desc.to_csv(os.path.join(a.out, "affective_engagement_by_influencer.csv"), index=False)
    print("\n[ranked affective-engagement rate, cluster-bootstrap 95% CI]")
    print(desc.assign(**{c: (100 * desc[c]).round(1) for c in
                         ["affective_rate", "ci_low", "ci_high",
                          "positive_rate", "negative_rate"]})[
          ["group", "handle", "n_posts", "n_comments", "affective_rate",
           "ci_low", "ci_high", "positive_rate", "negative_rate"]].to_string(index=False))

    res = {}
    aivi = eng[eng.group == "AIVI"]
    hi = eng[eng.group == "HI"]

    # ---- (1) heterogeneity within each group -----------------------------
    print("\n[test] between-influencer heterogeneity")
    for name, d in [("AIVI", aivi), ("HI", hi)]:
        kw, post_rates = kruskal_post_level(d, "affect")
        res[f"kruskal_{name}"] = kw
        print(f"  {name}: Kruskal-Wallis H={kw['H']:.1f}, df={kw['df']}, "
              f"p={kw['p']:.3g}, epsilon^2={kw['epsilon_squared']:.3f}")
        post_rates.to_csv(os.path.join(a.out, f"post_level_rates_{name}.csv"), index=False)
        try:
            g = gee_test(d, "affect")
            res[f"gee_{name}"] = g
            print(f"        GEE logistic Wald chi2={g['statistic']:.1f}, "
                  f"df={g['df']}, p={g['p']:.3g}  "
                  f"[{g['n_comments']} comments, {g['n_posts']} posts]")
        except Exception as e:
            print(f"        [GEE unavailable: {type(e).__name__}]")

    for outcome in ["pos", "neg"]:
        kw, _ = kruskal_post_level(aivi, outcome)
        res[f"kruskal_AIVI_{outcome}"] = kw
        print(f"  AIVI, {outcome:3s}: H={kw['H']:.1f}, df={kw['df']}, "
              f"p={kw['p']:.3g}, epsilon^2={kw['epsilon_squared']:.3f}")

    # ---- (2) is the AIVI spread wider than the HI spread? ----------------
    sp = spread_comparison(desc)
    pd.DataFrame([sp]).to_csv(os.path.join(a.out, "spread_comparison.csv"), index=False)
    print("\n[spread] dispersion of influencer-level rates")
    print(f"  AIVI SD={100*sp['AIVI_sd']:.1f} pp, range "
          f"{100*sp['AIVI_range_low']:.1f}-{100*sp['AIVI_range_high']:.1f}%, "
          f"IQR={100*sp['AIVI_IQR']:.1f} pp")
    print(f"  HI   SD={100*sp['HI_sd']:.1f} pp, range "
          f"{100*sp['HI_range_low']:.1f}-{100*sp['HI_range_high']:.1f}%, "
          f"IQR={100*sp['HI_IQR']:.1f} pp")
    print(f"  SD difference={100*sp['sd_difference']:+.1f} pp, 95% CI "
          f"[{100*sp['sd_diff_CI_low']:+.1f}, {100*sp['sd_diff_CI_high']:+.1f}]; "
          f"Fligner-Killeen chi2={sp['fligner_chi2']:.2f}, p={sp['fligner_p']:.3f}")
    print("\n[dispersion of valence components, SD across influencers]")
    for g in ["AIVI", "HI"]:
        s = desc[desc.group == g]
        print(f"  {g:4s} positive SD={100*s.positive_rate.std(ddof=1):.1f} pp, "
              f"negative SD={100*s.negative_rate.std(ddof=1):.1f} pp")
        for c in ["positive_rate", "negative_rate"]:
            rho, p = stats.spearmanr(s.affective_rate, s[c])
            print(f"       affective rate vs {c:14s} rho={rho:+.3f}  p={p:.4f}")

    pd.DataFrame(res).T.to_csv(os.path.join(a.out, "heterogeneity_tests.csv"))

    # ---- (3) observable characteristics -----------------------------------
    chars = characteristics(df, eng)
    m = desc.merge(chars, on=["group", "handle"])
    m.to_csv(os.path.join(a.out, "influencer_characteristics.csv"), index=False)
    cols = ["followers", "mean_likes_per_post", "mean_likes_per_comment",
            "comments_per_post", "mean_comment_length",
            "distinct_commenter_share", "posts_per_month", "artificiality_salience",
            "account_age_months"]
    corr = pd.concat([correlate(m, cols, "AIVI", "AIVI"),
                      correlate(m, cols, "HI", "HI")], ignore_index=True)
    corr.to_csv(os.path.join(a.out, "covariate_correlations.csv"), index=False)
    for lab in ["AIVI", "HI"]:
        print(f"\n[covariates] Spearman vs affective-engagement rate, {lab}")
        for _, r in corr[corr.subset == lab].iterrows():
            print(f"  {r.characteristic:26s} n={r.n:2.0f}  rho={r.spearman_rho:+.3f}  "
                  f"p={r.p:.4f}  p_bonf={r.p_bonferroni:.4f}")

    # ---- account age, AIVI vs HI -----------------------------------------
    age_a = m.loc[m.group == "AIVI", "account_age_months"].values.astype(float)
    age_h = m.loc[m.group == "HI", "account_age_months"].values.astype(float)
    u, pv = stats.mannwhitneyu(age_a, age_h, alternative="two-sided")
    delta = np.sign(age_a[:, None] - age_h[None, :]).mean()
    rng = np.random.default_rng(SEED)
    bs = [np.sign(rng.choice(age_a, len(age_a), True)[:, None]
                  - rng.choice(age_h, len(age_h), True)[None, :]).mean()
          for _ in range(N_BOOT)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    print("\n[account age] months since the account was opened")
    print(f"  AIVI median {np.median(age_a):.0f} (IQR {np.percentile(age_a,25):.0f}"
          f"-{np.percentile(age_a,75):.0f}); HI median {np.median(age_h):.0f} "
          f"(IQR {np.percentile(age_h,25):.0f}-{np.percentile(age_h,75):.0f})")
    print(f"  Mann-Whitney U={u:.1f}, p={pv:.3g}, Cliff's delta={delta:+.3f} "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    pd.DataFrame([{"AIVI_median": np.median(age_a), "HI_median": np.median(age_h),
                   "mannwhitney_U": u, "p": pv, "cliffs_delta": delta,
                   "CI_low": lo, "CI_high": hi}]).to_csv(
        os.path.join(a.out, "account_age_comparison.csv"), index=False)

    ac = age_correlations(m)
    ac.to_csv(os.path.join(a.out, "age_correlations.csv"), index=False)
    print("\n[account age] Spearman vs expression rates, with bootstrap CI and leave-one-out")
    for _, r in ac.iterrows():
        print(f"  {r.group:4s} {r.outcome:15s} n={r.n:2.0f}  rho={r.spearman_rho:+.3f}  "
              f"95% CI [{r.CI_low:+.3f}, {r.CI_high:+.3f}]  p={r.p:.4f}  "
              f"LOO [{r.loo_min:+.3f}, {r.loo_max:+.3f}]")

    print("\n[artificiality salience] mean % of comments referring to artificiality")
    print(m.groupby("group").artificiality_salience.describe()[
        ["count", "mean", "min", "50%", "max"]].mul(
        [1, 100, 100, 100, 100]).round(2).to_string())


if __name__ == "__main__":
    main()

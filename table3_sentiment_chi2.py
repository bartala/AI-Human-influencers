#!/usr/bin/env python3
"""
Table 3 significance tests ("Distribution of Sentiment Categories in User
Comments"): the omnibus chi-square test cited in the Results text, and each
category's own chi-square test (category vs. the other two categories
combined), all computed with the same single procedure so the three P-value
column entries in Table 3 come from one consistent method.

data_analysis.R computes the counts in Table 3 but no chi-square test appears
anywhere in the repository's R scripts (verified by grep across
data_analysis.R, model.R, graph_analysis.R for chisq.test / chi.test /
prop.test / fisher.test / p.adjust / bonferroni -- zero matches). This script
fills that gap using the same discrete sentiment label already committed in
Data/sentiment_analysis_results1.csv.zip (verified elsewhere to reproduce
Table 3's counts).

Run from the repository root: python3 table3_sentiment_chi2.py
Reads Data/sentiment_analysis_results1.csv.zip.
Writes Data/table3_chi2_results.csv.
"""
import io
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

SENT_CSV_ZIP = "Data/sentiment_analysis_results1.csv.zip"
OUT_CSV = "Data/table3_chi2_results.csv"
TYPE_MAP = {"AI": "AIVI", "HUMAN": "HI"}


def fmt_p(p):
    """chi2_contingency underflows to exactly 0.0 for very large statistics;
    report a floor consistent with double-precision limits rather than a
    bare 0.0."""
    return "< 1e-300" if p == 0.0 else f"{p:.4g}"


def main():
    with zipfile.ZipFile(SENT_CSV_ZIP) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        raw = z.read(name)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding="utf-8", encoding_errors="replace")
    df = df[["user_type", "sentiment"]].copy()
    df["type"] = df["user_type"].map(TYPE_MAP)

    a = df[df["type"] == "AIVI"]
    h = df[df["type"] == "HI"]
    na, nh = len(a), len(h)
    print(f"n AIVI = {na}, n HI = {nh}")

    counts_a = {v: (a["sentiment"] == v).sum() for v in [-1, 0, 1]}
    counts_h = {v: (h["sentiment"] == v).sum() for v in [-1, 0, 1]}
    print(f"AIVI negative/neutral/positive: {counts_a[-1]}/{counts_a[0]}/{counts_a[1]}")
    print(f"HI   negative/neutral/positive: {counts_h[-1]}/{counts_h[0]}/{counts_h[1]}")

    # omnibus 2x3 test, cited in the Results text
    tbl2x3 = np.array([[counts_a[-1], counts_a[0], counts_a[1]],
                        [counts_h[-1], counts_h[0], counts_h[1]]])
    chi2_omni, p_omni, dof_omni, _ = stats.chi2_contingency(tbl2x3)
    print(f"\nomnibus 2x3: chi2({dof_omni}) = {chi2_omni:.1f}, p {fmt_p(p_omni)}")

    # per-category tests, category vs. the other two combined -- the same
    # method for all three rows, matching Table 3's "chi2 test" footnote
    rows = []
    labels = {-1: "negative", 0: "neutral", 1: "positive"}
    print("\nper-category (category vs. rest), single consistent method:")
    for val, label in labels.items():
        ca, ch = counts_a[val], counts_h[val]
        tbl = np.array([[ca, na - ca], [ch, nh - ch]])
        chi2, p, dof, _ = stats.chi2_contingency(tbl)
        print(f"  {label:9s}: chi2({dof}) = {chi2:.1f}, p {fmt_p(p)}")
        rows.append({"category": label, "AIVI_n": ca, "HI_n": ch,
                      "chi2": chi2, "p": p, "p_display": fmt_p(p)})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()

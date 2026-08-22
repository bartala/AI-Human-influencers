# Analyses of comments classified as negative

Analyses of the 11,882 comments assigned negative polarity (AIVI n = 4,396;
HI n = 7,486), with the influencer as the unit of analysis.

## Scripts

| Script | Purpose |
|---|---|
| `mention_analysis.py` | @mention indicators; within-influencer and between-group comparisons |
| `artificiality_keywords.py` | Artificiality term list; between-group and within-AIVI tests; per-term counts; all-35 and English-only sensitivities |
| `make_figure.py` | Figure 4 (run after the two analysis scripts) |

## Usage

```bash
pip install -r requirements.txt
python mention_analysis.py       --data path/to/sentiment_analysis_results1.csv
python artificiality_keywords.py --data path/to/sentiment_analysis_results1.csv
python make_figure.py
```

Outputs are written to `output/`. Seed fixed at 42; 10,000 bootstrap resamples.
Input comment counts are validated against Table 3 on startup.

## Per-term counts in comments classified as negative

Reproduced by `artificiality_keywords.py`
(`output/artificiality_term_breakdown.csv`).

| Term | AIVI | HI |
|---|---|---|
| robot(s) | 217 | 0 |
| ai | 171 | 1 |
| not real / not human / is she real | 38 | 3 |
| virtual | 15 | 2 |
| artificial | 12 | 2 |
| cgi | 12 | 0 |
| bot(s) | 11 | 0 |
| computer-/AI-generated | 7 | 0 |
| deepfake | 5 | 0 |
| avatar | 3 | 1 |
| a.i. | 2 | 0 |
| android/cyborg | 2 | 0 |

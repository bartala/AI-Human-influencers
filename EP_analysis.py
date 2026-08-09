import pandas as pd
from scipy.stats import mannwhitneyu

# Earnings per post data
aivi_ep = [1700, 249, 1700, 1000, 146, 263, 262, 1800, 249, 252, 2300, 2000, 24700, 1200,
           7600, 2000, 291, 161, 263, 1100, 510, 2100, 232]
hi_ep = [5000, 3000, 24400, 3200, 1300, 5700, 10500, 302, 1300, 556, 1000, 22200]

# Perform Mann-Whitney U test (non-parametric test for difference in medians)
stat, p_value = mannwhitneyu(aivi_ep, hi_ep, alternative='two-sided')

# Prepare summary
aivi_median = pd.Series(aivi_ep).median()
hi_median = pd.Series(hi_ep).median()
aivi_mean = pd.Series(aivi_ep).mean()
hi_mean = pd.Series(hi_ep).mean()

result = {
    "AIVI mean EP": aivi_mean,
    "HI mean EP": hi_mean,
    "AIVI median EP": aivi_median,
    "HI median EP": hi_median,
    "Mann-Whitney U statistic": stat,
    "p-value": p_value
}

result


# Earnings per post data
aivi_ep = [1700, 249, 1700, 1000, 146, 263, 262, 1800, 249, 252, 2300, 2000, 24700, 1200,
           7600, 2000, 291, 161, 263, 1100, 510, 2100, 232]
hi_ep = [5000, 3000, 24400, 3200, 1300, 5700, 10500, 302, 1300, 556, 1000, 22200]

# Perform Mann-Whitney U test (non-parametric test for difference in medians)
stat, p_value = mannwhitneyu(aivi_ep, hi_ep, alternative='two-sided')

# Prepare summary
aivi_median = pd.Series(aivi_ep).median()
hi_median = pd.Series(hi_ep).median()
aivi_mean = pd.Series(aivi_ep).mean()
hi_mean = pd.Series(hi_ep).mean()

result = {
    "AIVI mean EP": aivi_mean,
    "HI mean EP": hi_mean,
    "AIVI median EP": aivi_median,
    "HI median EP": hi_median,
    "Mann-Whitney U statistic": stat,
    "p-value": p_value
}

result

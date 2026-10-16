# `sl_feature_comparison_tools.py`

Selection now also saves RFECV curves with fold variability, wrapper permutation
importance, embedded importance, grid-search diagnostics, PCA feature ranking
and a Shapiro p-value histogram. See [Result Plots](result_plots.md).

Reusable implementation library behind SL feature comparison. It contains data-cleaning helpers, Shapiro tests, ANOVA, Fisher score, mutual information, PCA ranking, plotting, RFECV wrappers, and embedded feature-importance methods.

`sl_feature_comparison.py` imports these functions to control the experiment. This module is not normally run as a standalone pipeline entrypoint.

See `sl_feature_comparison_tools_guide.md` for a detailed description of the methods.

Legacy columns match only `f` followed by digits, such as `f0` or `f20`.
Descriptive Fourier columns (`fourier_1` through `fourier_20`) remain eligible
for selection; they are not discarded simply because their names start with f.

Saved figures use Matplotlib's non-interactive `Agg` backend. No Tkinter plot
windows are created, avoiding GUI cleanup conflicts when `sl_main.py` moves from
feature comparison into threaded model training. Ranking and selection logic
are unchanged.

# `sl_feature_comparison_tools.py`

Reusable implementation library behind SL feature comparison. It contains data-cleaning helpers, Shapiro tests, ANOVA, Fisher score, mutual information, PCA ranking, plotting, RFECV wrappers, and embedded feature-importance methods.

`sl_feature_comparison.py` imports these functions to control the experiment. This module is not normally run as a standalone pipeline entrypoint.

See `sl_feature_comparison_tools_guide.md` for a detailed description of the methods.

Saved figures use Matplotlib's non-interactive `Agg` backend. No Tkinter plot
windows are created, avoiding GUI cleanup conflicts when `sl_main.py` moves from
feature comparison into threaded model training. Ranking and selection logic
are unchanged.

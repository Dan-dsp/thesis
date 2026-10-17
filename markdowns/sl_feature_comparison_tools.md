# `sl_feature_comparison_tools.py`

PCA now takes `seed=42` and passes it to `PCA(random_state=seed)`; MI also accepts
the workflow seed. Augmentation identifiers are metadata, never features, and
are preserved in reduced exports. Rankings/diagnostics still use original train
rows only. See [Shared Augmentation](shared_augmentation.md).

The project workflow passes a shared image manifest into `load_features_and_labels`.
Only its selected train rows are used, ordered by DL image IDs rather than CSV order;
test rows are retained for export. See [Shared Training Pool](shared_training_pool.md).

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

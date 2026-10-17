# thesis_validation.py

Nested SL CV now uses shared augmented descriptors only within fitting folds;
inner and outer held-out folds contain originals only. Original-image partitions
are selected at runtime, not preassigned. Selection stays inside inner folds.
See [Shared Augmentation](shared_augmentation.md).

SL nested CV now starts from the same ordered balanced development pool as DL,
and saves its training_pool.csv/JSON and pool ID. Statistical comparisons refuse
different pool IDs when present. See [Shared Training Pool](shared_training_pool.md).

Nested CV saves outer-fold confusion matrices and `outer_cv_scores.png`.
The within-family comparison command also plots the independent fold scores.
See [Result Plots](result_plots.md); these plots do not replace the statistical tests.

The sl-cv command runs nested SL evaluation with MI or ANOVA selected inside inner folds.
The compare command tests paired outer macro-F1 scores within one family using a corrected
paired t-test and Holm correction. It refuses cross-family tests and invalid fold pairings.

See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md) for assumptions and commands.

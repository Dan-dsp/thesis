# thesis_validation.py

The sl-cv command runs nested SL evaluation with MI or ANOVA selected inside inner folds.
The compare command tests paired outer macro-F1 scores within one family using a corrected
paired t-test and Holm correction. It refuses cross-family tests and invalid fold pairings.

See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md) for assumptions and commands.

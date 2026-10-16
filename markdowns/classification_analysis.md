# classification_analysis.py

Shared prediction-audit and rejection helpers for SL and DL; also a CLI for descriptive
image-error overlap and manually annotated failure conditions. Rejection helpers remain
unchanged; the error report uses raw species predictions, not acceptance decisions.

The CLI now delegates to [Image Error Analysis](image_error_analysis.md), which saves
a visual image/model error grid, local HTML gallery, shared/complementary failures,
species confusion summaries and optional condition rates. `--image-root` resolves
images from their species/filename keys. No individual-bird IDs are required.

See [Thesis Evaluation Protocol](thesis_evaluation_protocol.md) for outputs and usage.

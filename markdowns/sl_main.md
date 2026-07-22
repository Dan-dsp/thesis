# `sl_main.py`

Top-level SL orchestrator. It can run feature extraction, feature comparison, and training in sequence, while allowing each stage to be turned on or off through its `main()` arguments.

The stages are `sl_dataframe_main.py`, `sl_feature_comparison.py`, and `sl_training_pipeline.py`. Use this script for an end-to-end SL run; use the individual scripts when you want to repeat just one stage.

See `sl_main_guide.md` for the detailed guide.

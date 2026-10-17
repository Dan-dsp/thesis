# Top-Level Python File Index

This index covers only `.py` files directly inside `01_Python`.

| File | Role | Pipeline status |
| --- | --- | --- |
| `split_dataset.py` | Creates train/test image folders | Dataset preparation |
| `rename_images_from_excel.py` | Renames image files from an Excel list | Dataset preparation |
| `dl_transfer_learning_bird_classifier.py` | Full ResNet18 training and test evaluation | DL training |
| `dl_transfer_learning_bird_classifier_partial_finetune.py` | ResNet18 partial-fine-tuning experiment | DL training alternative |
| `predict_single_image.py` | Classifies one image with a saved checkpoint | DL inference helper |
| `predict_interactive.py` | Chooses an image in a desktop dialog and predicts it | DL inference helper |
| `dl_main.py` | Detection, crop preparation, classification, annotation | DL inference entrypoint; currently has missing local dependencies |
| `compare_dl_sl_models.py` | Writes independent SL and DL reports | Family evaluation |
| `classification_analysis.py` | Image/bird error audits and validation-selected rejection | Shared analysis |
| `image_error_analysis.py` | Shared-image error matrices, gallery, species and condition summaries | Descriptive image analysis |
| `thesis_validation.py` | Nested SL CV and within-family outer-fold tests | Thesis validation |
| `result_plots.py` | Non-GUI PNGs from existing selection/CV/evaluation tables | Shared visualization |
| `predict_sl_with_rejection.py` | Predicts features with the selected SL model and rejection | SL inference |
| `sl_dataframe_main.py` | Configures handcrafted-feature dataset construction | SL entrypoint |
| `sl_dataframe_construction.py` | Reads images and writes feature CSV outputs | SL feature extraction |
| `sl_methods.py` | Defines handcrafted image descriptors | SL feature library |
| `sl_feature_comparison.py` | Coordinates feature selection and exports reduced datasets | SL feature selection |
| `sl_feature_comparison_tools.py` | Implements selection, ranking, and diagnostic methods | SL feature-selection library |
| `sl_training_pipeline.py` | Trains and evaluates classical ML models | SL training |
| `sl_main.py` | Runs selected SL stages in order | SL orchestrator |
| `run_pipeline.py` | Sequential SL/DL training and reporting launcher with saved logs | Pipeline orchestration |
| `shared_training_pool.py` | DL-ordered balanced image list reused by all SL/DL training stages | Shared training population |
| `shared_augmentation.py` | Fixed image views shared by SL/DL; runtime fitting-only expansion | Shared augmentation |
| `sl_models_evaluation.py` | Extra plotting and evaluation utilities | Older SL helper |

The current end-to-end SL route is `sl_dataframe_main.py` -> `sl_feature_comparison.py` -> `sl_training_pipeline.py`, optionally launched by `sl_main.py`.

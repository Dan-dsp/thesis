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
| `compare_dl_sl_models.py` | Evaluates saved DL and SL models on comparable data | Cross-pipeline evaluation |
| `sl_dataframe_main.py` | Configures handcrafted-feature dataset construction | SL entrypoint |
| `sl_dataframe_construction.py` | Reads images and writes feature CSV outputs | SL feature extraction |
| `sl_methods.py` | Defines handcrafted image descriptors | SL feature library |
| `sl_feature_comparison.py` | Coordinates feature selection and exports reduced datasets | SL feature selection |
| `sl_feature_comparison_tools.py` | Implements selection, ranking, and diagnostic methods | SL feature-selection library |
| `sl_training_pipeline.py` | Trains and evaluates classical ML models | SL training |
| `sl_main.py` | Runs selected SL stages in order | SL orchestrator |
| `sl_models_evaluation.py` | Extra plotting and evaluation utilities | Older SL helper |

The current end-to-end SL route is `sl_dataframe_main.py` -> `sl_feature_comparison.py` -> `sl_training_pipeline.py`, optionally launched by `sl_main.py`.

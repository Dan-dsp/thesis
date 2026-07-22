# `predict_single_image.py`

Provides `predict_image(model_path, image_path)` for straightforward DL inference on one already-cropped bird image. It restores a saved ResNet18 checkpoint, applies the required resize and normalization, then prints the most likely species and confidence.

The bottom of the file contains an example invocation with paths to edit. It is useful for checking a trained checkpoint manually, but is not invoked by the DL training scripts.

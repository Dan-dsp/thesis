# `sl_methods.py`

Handcrafted-feature library for the SL pipeline. It converts an image tensor into a fixed feature vector using shape, contour, frequency, color, texture, gradient, and optional visual-word descriptors.

`extract_all_features_torch()` is the main public function and `get_all_feature_names()` keeps its column names aligned. `sl_dataframe_construction.py` calls this module for every input image.

See `sl_methods_guide.md` for descriptor-level explanations.

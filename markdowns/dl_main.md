# `dl_main.py`

This is the older full-image DL inference entrypoint. Given an image and a mode (`mask` or bounding box), it detects birds, creates classifier-ready crops, predicts each crop’s species, and annotates the original image.

It calls `bird_detector`, `utils_bounding_boxes_separation`, and `predict_image_from_tensors`. Those local modules are not currently present at the top level, so the script will not run until its missing detector and crop/prediction dependencies are restored or its imports are updated.

It is an inference/deployment script only. It does not train a model and is unrelated to the SL pipeline.

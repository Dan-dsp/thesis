# `predict_interactive.py`

An interactive, single-image DL predictor. It opens a file-selection dialog, loads a ResNet18 checkpoint, preprocesses the chosen image, prints the predicted species and confidence, and displays the image.

The checkpoint path is hard-coded in `main()` and must be updated for the current machine/model. This is a convenience inference helper, not part of training and not used by the SL pipeline.

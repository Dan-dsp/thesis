# `dl_transfer_learning_bird_classifier_partial_finetune.py`

This is an alternative DL training experiment based on the same ResNet18/ImageFolder approach as `dl_transfer_learning_bird_classifier.py`. Its distinguishing choice is partial fine-tuning: it controls which backbone layers are trainable rather than always fine-tuning the whole network.

It performs internal validation, optional K-fold reporting, final training, and test evaluation, producing checkpoints and plots. Use it to compare a more constrained fine-tuning strategy against the main training script; it is not required by the prediction pipeline.

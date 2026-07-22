# `split_dataset.py`

Dataset-preparation script that copies each species folder into a reproducible `train` and `test` ImageFolder layout. It shuffles each species with a seed, then applies the configured ratio independently per species.

The current default split is 90% training and 10% testing. There is no validation folder: the former validation share was moved into training, and the DL/SL training scripts create validation internally when required.

Run it before DL or SL training. It is a preparatory step, not a model pipeline stage itself.

# predict_sl_with_rejection.py

Loads selected_model.json, the saved SL model, encoder and ordered feature columns, then
predicts a feature CSV with the validation-selected rejection threshold.

```powershell
python predict_sl_with_rejection.py path/to/selected_model.json path/to/features.csv --output predictions.csv
```

The CSV must contain the selected model's exact handcrafted features. Output includes
predicted_label, confidence, rejected and decision (species or REJECT).

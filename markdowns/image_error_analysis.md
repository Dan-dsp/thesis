# `image_error_analysis.py`

Descriptive image-error reporting, without rejection or individual-bird tracking.
The library is used automatically by `compare_dl_sl_models.py`, and through the
existing `classification_analysis.py` command. It never fits a classifier.

## Automatic Report

After corrected SL selection/training, run from `01_Python`:

```powershell
python compare_dl_sl_models.py
```

The full-DL trainer now defaults to `outputs_resnet18_thesis`, while your existing
full-DL checkpoint is in `outputs_resnet18_thesis_dl_bbox`. To reuse that saved
checkpoint explicitly without retraining:

```powershell
python compare_dl_sl_models.py --dl-full-model outputs_resnet18_thesis_dl_bbox/final_train/bird_species_resnet18_FINAL_best.pth
```

`--dl-partial-model` can override the partial-fine-tuning checkpoint too. The
report does not automatically substitute a historical checkpoint for a new one.

The script evaluates the two saved DL checkpoints and four SL classifiers from
the chosen SL run. All test images are included in prediction audits, even if
`--balance-test` requests balanced aggregate performance tables. No confidence
threshold or rejected/accepted decision is used for this analysis.

Open `family_evaluation_results/image_error_analysis/error_gallery.html` locally.
Each error image is shown with its true species and every available model's raw
prediction. Images are loaded from local disk without modifying the source files;
keep the report on this computer or supply the image files when moving it.

## Report Contents

| File | Information |
| --- | --- |
| `all_predictions.csv` | Every evaluated image/model prediction, including correct predictions |
| `error_review.csv` | Wrong raw species predictions, regardless of rejection flags in older audits |
| `both_families_error.csv` | Images with at least one wrong SL prediction and at least one wrong DL prediction |
| `sl_only_error.csv` / `dl_only_error.csv` | Images evaluated by both families but wrong only within one family |
| `unmatched_images.csv` | Images not evaluated by both families; never interpreted as successes in the missing family |
| `image_error_frequency.csv` | Per-image SL/DL evaluation and error counts, true species, local path, category |
| `image_error_matrix.csv` | Image-by-evaluator grid: 1 wrong, 0 correct, blank not evaluated |
| `image_prediction_matrix.csv` | Predicted species for each image/evaluator |
| `image_errors.png` | Visual error grid for up to 60 images with the most errors; all errors remain in CSV/gallery |
| `descriptive_error_overlap.csv` | Pairwise shared errors, complementary successes, coverage and same wrong-species counts |
| `species_error_summary.csv` | Per-classifier error counts/rates by true species |
| `confusion_pairs.csv` | Which wrong species each classifier assigns and how often |
| `evaluator_sources.csv` | Identifies the source file for each model/run; repeated model names stay distinct |
| `error_summary.json` | Coverage, missing image-file count, shared-error count and interpretation limits |
| `annotation_template.csv` | Optional per-image conditions and notes; no individual-bird ID needed |
| `condition_summary.csv` | Annotated-condition sample counts, errors and error rates per model/run |

An error in both families does not mean every model failed. Family-level counts
also depend on how many classifiers are included. Use the per-model
grid and predictions to distinguish individual classifiers. This is descriptive
overlap inspection, not a joint SL/DL ranking or cross-family significance test.

## Image Correspondence

Images match only by the exact `sample_key` (`species/original_filename`). Same
species alone is not sufficient. Different filenames/crops are not fuzzy-matched;
verify correspondence before merging them. Conflicting true labels or duplicate
image/model records are refused. Missing local image files do not discard CSV
predictions; the gallery explicitly marks the image unavailable.

## Common Conditions

Review images in the gallery and annotate conditions such as
`blur;occlusion;backlighting;small_bird;crop_truncated` in the template. Review
successful images too, using `all_predictions.csv` and local images, so condition
error rates have meaningful denominators. Preserve empty conditions for images
not yet reviewed; use `clear` for reviewed clear images if appropriate.

To rebuild from saved audits with annotations, without inference or retraining:

```powershell
python classification_analysis.py path/to/sl_test_predictions.csv path/to/dl_test_predictions.csv --out-dir error_review --image-root ../dataset_bbox/test --annotations path/to/annotations.csv
```

You may pass more than two prediction files. Include all desired classifiers.
An existing template is not overwritten by a blank template. Record observed
conditions, not assumed causes; an association does not establish causation.
Inspecting crops can suggest cropping issues, but attributing failures to the
detector requires reviewing the corresponding original photographs as well.

Existing rejection functions remain unchanged and separate. Historical
`bird_id` annotations remain readable for compatibility, but are not requested
by the current image-focused template.

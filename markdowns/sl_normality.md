# Within-Species Normality Diagnostics

`sl_normality.py` is called through `run_shapiro_tests` in the normal SL feature
comparison workflow. It can also run independently without feature selection
or model retraining:

```powershell
# From F:\01_Univalle\01_TG
python 01_Python/sl_normality.py
```

It loads the existing descriptor CSV and shared balanced pool. Only original
`train` rows are examined: no test rows or augmented views. With the current
pool, each feature is tested separately in each of 12 species (95 originals per
species before dropping nonfinite values). 313 features yield 3,756 tests.

## Assessment

Shapiro-Wilk uses alpha 0.05 and a deterministic seed-42 subsample if a
feature/species pair exceeds 5,000 finite values. The current pool needs no
subsampling. A constant feature or fewer than three finite values is explicitly
`not_testable`, not evidence for or against normality. The output preserves
`gaussian_at_alpha` for compatibility, but `test_status` is the preferred field:
`rejected`, `not_rejected`, or `not_testable`.

P-values are unadjusted exploratory diagnostics across many tests; they are not
family-wise or false-discovery-controlled decisions. A high p-value does not
prove Gaussianity, and one marginal test does not establish multivariate
normality, independence or equal class variances. No features are removed and
no training transformations change as a consequence of these tests.

## Conservative Plot Selection

One feature per active descriptor family is reused across all species:
Hu, Zernike, Fourier, affine, log-polar, HSV histogram, color moments, LBP, HOG,
GLCM, Gabor and edge histogram. This gives 12 figures for the current extractor.
Optional SIFT/ORB families would add a figure only when present.

Within each family, choose the feature that is nonconstant in the largest number
of species; ties use the existing feature-column order. No p-values or model
performance are used to choose a visually favorable example. The saved
`representative_features.csv` makes the choice auditable. These examples do not
represent every feature's distribution; all features still receive tests.

Each PNG is a species-panel grid with the same feature and histogram bins/x-axis
across species, showing raw descriptor values, sample count, raw p-value and test
status. The red curve is a Gaussian with that species' sample mean and standard
deviation, not proof of a Gaussian fit. Values already transformed by the
extractor (for example Hu's log representation) are plotted as stored.

This examines **normality**, not **normalization**. There is no extra log,
z-score or Gaussianizing transformation of training features in this diagnostic.

## Saved Outputs

Under `sl_results/exploratory/shapiro/`:

- `shapiro_normality.csv`: one row per species/feature; finite sample counts,
  subsampling flag, W, raw p-value, test status and notes.
- `shapiro_by_species_summary.csv`: rejected/not-rejected/not-testable counts per species.
- `shapiro_summary.json`: scope, settings and total counts.
- `shapiro_p_values.png`: p-value distributions distinguished by species.
- `representative_features.csv`: fixed representative used for each family.
- `distributions/<family>_within_species.png`: the 12 distribution grids.

Running the diagnostics replaces the old pooled CSV/JSON/p-value plot at these
paths. Historical pooled counts (9/304) must not be described as current
within-species results. Nothing else in the training, selection or comparison
pipeline needs rerunning solely for this diagnostic change.

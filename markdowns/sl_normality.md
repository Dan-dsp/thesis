# Within-Species Normality Diagnostics

`sl_normality.py` is called through `run_shapiro_tests` in the normal SL feature
comparison workflow. It can also run independently without feature selection
or model retraining:

```powershell
# From F:\01_Univalle\01_TG
python 01_Python/sl_normality.py
```

Clicking Run also works without arguments. The default input is
`dataset_features/shallow_learning_birds.csv`, resolved relative to the script's
project root, not the terminal's working directory. `--data-csv` remains an
optional override. It loads the CSV and shared balanced pool. Only original
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

## Highest and Lowest Normality Plots

Two features per active descriptor family are selected and reused across all species:
Hu, Zernike, Fourier, affine, log-polar, HSV histogram, color moments, LBP, HOG,
GLCM, Gabor and edge histogram. This gives up to 24 figures for the current extractor.
Optional SIFT/ORB families add figures only when present and testable.

Within each family, compute each feature's arithmetic mean Shapiro-Wilk W across
species, weighting each species equally. Select the largest mean W as `highest`
and the smallest as `lowest`. W closer to 1 indicates closer agreement with
normality; it is not a probability. This aggregate ranking is a visualization
convention, not a new statistical test, and does not imply the highest-ranked
feature passes the p-value threshold in every species.

Only features with valid W and p-values in every species are eligible. Constant,
insufficient-data or numerically untestable distributions cannot be ranked as
either extreme. Ties follow existing feature-column order. If there is only one
eligible feature, both plots use that feature; if none, the manifest records
`no_fully_testable_feature` and no plot is created for that family.

`representative_features.csv` records the extremes, mean W, eligible-feature
counts and selection rule. These are deliberately selected best/worst examples,
not typical examples of the whole family. All features still receive tests, and
their individual p-values remain unadjusted exploratory results.

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
- `representative_features.csv`: highest/lowest feature per family and mean W.
- `distributions/normality_extremes/<family>_highest_normality.png`: highest mean W.
- `distributions/normality_extremes/<family>_lowest_normality.png`: lowest mean W.

Old `distributions/<family>_within_species.png` files are retained as historical
outputs; they are not the newly selected extremes. Use the manifest and the
`normality_extremes` subfolder for the current figures.

Running the diagnostics replaces the old pooled CSV/JSON/p-value plot at these
paths. Historical pooled counts (9/304) must not be described as current
within-species results. Nothing else in the training, selection or comparison
pipeline needs rerunning solely for this diagnostic change.

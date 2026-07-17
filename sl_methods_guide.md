# `sl_methods.py` Guide

This document explains what each function in [sl_methods.py](f:/01_Univalle/01_TG/01_Python/sl_methods.py) is doing, the theory behind the descriptor, and whether the implementation follows that theory closely.

The goal is that you can read this file first, understand the workflow of each method, and then read the code with much less friction.

## Big Picture

[sl_methods.py](f:/01_Univalle/01_TG/01_Python/sl_methods.py) is a **feature extraction library** for shallow learning. It converts one image tensor into a long handcrafted feature vector by concatenating several descriptor families:

- shape descriptors
- contour descriptors
- local texture descriptors
- color descriptors
- gradient descriptors
- frequency descriptors
- optional Bag-of-Visual-Words descriptors

The final function, `extract_all_features_torch(...)`, is the main entry point. Everything else is a descriptor-specific helper.

---

## Helper Functions

### `get_all_feature_names(...)`

#### Goal

Return the names of the features in the same order they are expected to appear in the final feature vector.

#### Workflow

1. Add Hu names.
2. Add Zernike names.
3. Add Fourier descriptor names.
4. Add affine-invariant names.
5. Add log-polar FFT names.
6. Add HSV histogram names.
7. Add color moments.
8. Add LBP names.
9. Add HOG names.
10. Add GLCM names.
11. Add Gabor names.
12. Add edge histogram names.
13. Add optional SIFT / ORB BoVW names.

#### Theory Check

This is not a descriptor itself; it is only naming logic.

#### Implementation Check

The function now keeps the optional `sift_bovw_*` and `orb_bovw_*` names at the very end, which matches `extract_all_features_torch(...)`.

So the naming order and the extraction order are now aligned.

---

### `_to_numpy_gray(img_t)` and `_to_numpy_bgr(img_t)`

#### Goal

Convert a PyTorch image tensor `(C,H,W)` in `[0,1]` into OpenCV-compatible numpy images.

#### Workflow

- `_to_numpy_gray(...)`:
  1. move tensor to CPU
  2. clamp to `[0,1]`
  3. convert to `uint8`
  4. convert RGB to BGR
  5. convert BGR to grayscale

- `_to_numpy_bgr(...)`:
  1. move tensor to CPU
  2. clamp to `[0,1]`
  3. convert to `uint8`
  4. convert RGB to BGR

#### Theory Check

These are preprocessing utilities and they are correct for OpenCV usage.

---

## Shape and Contour Descriptors

### `extract_hu_moments_torch(img_t)`

#### Goal

Compute the 7 Hu invariant moments.

#### Theory

Hu moments are nonlinear combinations of normalized central moments designed to be invariant to:

- translation
- scale
- rotation

In many shape-analysis workflows, Hu moments are computed from a binary silhouette or segmented object.

#### Workflow in Code

1. Convert image to grayscale.
2. Compute spatial moments with `cv2.moments(...)`.
3. Compute Hu moments with `cv2.HuMoments(...)`.
4. Apply signed log transform.

#### Theory Check

Yes, the implementation follows the standard Hu-moment workflow.

#### Caveat

The code computes Hu moments from the **grayscale image**, not from a strict binary mask. That is still valid mathematically, but it behaves more like an intensity-weighted moment descriptor than a pure silhouette descriptor.

---

### `extract_zernike_moments_torch(img_t, radius=64, degree=6)`

#### Goal

Compute Zernike moments for a fixed order `n = degree`.

#### Theory

Zernike moments are orthogonal moments defined on the unit disk. They are widely used for shape description because they provide compact, rotation-aware shape information.

A full Zernike descriptor often includes **multiple orders and repetitions**. This implementation uses only a **single degree** and collects all valid `m` values for that degree.

#### Workflow in Code

1. Convert image to grayscale.
2. Resize to a fixed square grid around a disk.
3. Binarize with Otsu thresholding.
4. Build radius `r` and angle `theta`.
5. Keep only pixels inside the unit disk.
6. Compute the radial polynomial `R_n^m(r)`.
7. Compute each `Z_n^m`.
8. Return the magnitudes.

#### Theory Check

Yes, the core Zernike formula is implemented correctly as a discrete approximation.

#### Caveats

- This is a **partial Zernike descriptor**, not a full multi-order set.
- The use of a fixed binarized image is appropriate for shape, but it assumes the object is reasonably segmented.
- The descriptor uses only magnitudes, which is a common way to reduce rotation dependence.

---

### `extract_fourier_descriptors_torch(img_t, k=20)`

#### Goal

Compute contour-based Fourier descriptors.

#### Theory

Fourier descriptors represent a contour as a complex 1D signal and analyze its frequency content with the FFT.

Typical invariance steps:

- translation invariance: subtract the contour mean
- scale invariance: normalize by a reference magnitude
- optionally rotation invariance: use magnitudes rather than complex coefficients

#### Workflow in Code

1. Convert image to grayscale.
2. Detect edges with Canny.
3. Find contours.
4. Choose the largest contour.
5. Convert contour points into complex values `x + jy`.
6. Subtract the mean.
7. Apply FFT.
8. Drop the DC term.
9. Keep the first `k` magnitudes.
10. Normalize by the first magnitude if nonzero.

#### Theory Check

Yes, this matches the standard contour-Fourier-descriptor idea well.

#### Caveat

If the largest contour is not the bird contour, the descriptor will follow the wrong shape. That is a segmentation/edge-quality issue, not a theory issue.

---

### `extract_affine_invariants_torch(img_t)`

#### Goal

Compute affine-invariant combinations of image moments.

#### Theory

Affine moment invariants are designed to remain stable under affine transformations such as:

- translation
- scaling
- rotation
- shear

They are built from normalized central moments.

#### Workflow in Code

1. Convert image to grayscale.
2. Compute raw moments.
3. Compute centroid.
4. Compute central moments.
5. Normalize them into `eta(p,q)`.
6. Build 5 invariant combinations.

#### Theory Check

Yes, with the duplicated term removed, the implementation is now aligned with the intended distinct-invariant set it returns.

#### Implementation Check

The duplicated 6th invariant was removed.

So the function now returns **5 distinct affine invariants**, which is better aligned with the actual math than returning a repeated value.

---

## Frequency / Spectral Descriptor

### `extract_log_polar_fft_torch(img_t, out_radial=8, out_angular=8)`

#### Goal

Build a compact descriptor from the Fourier magnitude spectrum after log-polar remapping.

#### Theory

The usual intuition is:

- image translation affects phase more than magnitude
- Fourier magnitude captures frequency structure
- log-polar remapping can convert scale/rotation changes into translations in transformed coordinates

So a log-polar FFT descriptor tries to become more robust to rotation and scale.

#### Workflow in Code

1. Convert image to grayscale.
2. Convert it to float, remove the mean, and apply a Hanning window.
3. Compute 2D FFT.
4. Shift zero frequency to the center.
5. Compute log-magnitude.
6. Suppress the DC peak and robustly normalize the spectrum.
7. Apply OpenCV `logPolar(...)` with a scale derived from image size.
8. Replace non-finite values, resize to a fixed small grid, and L2-normalize the vector.

#### Theory Check

The implementation follows the standard idea of a log-polar FFT descriptor.

#### Implementation Check

This was the descriptor that had produced unstable values before. The implementation now adds several stabilization steps:

- mean removal before the FFT
- Hanning windowing to reduce border artifacts
- `log1p` magnitude instead of raw magnitude
- explicit `nan` / `inf` cleanup
- percentile-based normalization instead of trusting a single extreme bin
- radius-aware log-polar scaling instead of a fixed heuristic
- final L2 normalization of the descriptor

So the theory is still the same, but the implementation is now much more numerically robust in practice.

---

## Local Keypoint Descriptors

### `extract_sift_bovw_torch(img_t, kmeans_centers)`

#### Goal

Convert variable-length SIFT descriptors into a fixed-length Bag-of-Visual-Words histogram.

#### Theory

The BoVW pipeline is:

1. extract local descriptors
2. cluster descriptors into `K` visual words
3. assign each descriptor to its nearest cluster center
4. build a histogram of visual-word counts

This gives a fixed-length representation independent of the number of keypoints.

#### Workflow in Code

1. Convert image to grayscale.
2. Extract SIFT descriptors.
3. Compute Euclidean distance to all cluster centers.
4. Assign each descriptor to the nearest center.
5. Build a histogram.
6. L1-normalize it.

#### Theory Check

Yes, this is a standard SIFT + BoVW implementation.

#### Caveat

This is a simple BoVW variant:

- no TF-IDF
- no power normalization
- no spatial pyramid

That is fine if the goal is a compact baseline descriptor.

---

### `extract_orb_bovw_torch(img_t, kmeans_centers)`

#### Goal

Build a BoVW descriptor from ORB local descriptors.

#### Theory

ORB descriptors are binary descriptors. In standard ORB matching, the natural metric is **Hamming distance**, not Euclidean distance.

#### Workflow in Code

1. Convert image to grayscale.
2. Extract ORB descriptors.
3. Convert the codebook to byte patterns if needed.
4. Compute Hamming distance to the centers.
5. Assign nearest center.
6. Build and normalize the histogram.

#### Theory Check

Yes, the assignment step is now aligned with ORB theory.

#### Important Caveat

The assignment step now uses **Hamming distance**, which is the correct matching idea for ORB.

That said, one nuance remains:

- if your visual-word prototypes were originally trained with ordinary Euclidean k-means on float-cast ORB descriptors, then the **training-time codebook** is still only an approximation
- the current code rounds those prototypes back to `uint8` byte patterns before Hamming assignment

So the matching step is now theory-aligned, but the cleanest full pipeline would be:

- build the ORB codebook with a binary-aware clustering/prototype method
- then assign descriptors with Hamming distance

---

## Color Descriptors

### `extract_color_histogram_torch(img_t, bins=(4,4,4))`

#### Goal

Compute a 3D HSV color histogram.

#### Theory

Color histograms represent the distribution of colors in an image. HSV is often used because:

- hue captures chromatic identity
- saturation captures color purity
- value captures brightness

#### Workflow in Code

1. Convert RGB tensor to BGR.
2. Convert BGR to HSV.
3. Compute a 3D histogram with `cv2.calcHist(...)`.
4. L1-normalize.
5. Flatten to a vector.

#### Theory Check

Yes, this is a standard HSV histogram descriptor.

---

### `extract_color_moments_torch(img_t)`

#### Goal

Compute color moments for each RGB channel.

#### Theory

Color moments summarize each channel using:

- mean
- standard deviation
- skewness

This is a compact alternative to full histograms.

#### Workflow in Code

1. For each RGB channel:
   - compute mean
   - compute standard deviation
   - compute skewness
2. Concatenate all values.

#### Theory Check

Yes, this matches the standard color-moment idea.

#### Caveat

Skewness can become unstable when the channel variance is very small. The code handles that reasonably with a small epsilon in the denominator.

---

## Texture / Gradient Descriptors

### `extract_lbp_uniform_hist_torch(img_t, normalize=True)`

#### Goal

Compute a uniform LBP histogram with 59 bins.

#### Theory

Uniform LBP with `P=8`, `R=1` is a classic texture descriptor:

- each pixel is compared to its 8 neighbors
- this yields an 8-bit local binary code
- “uniform” patterns are those with at most 2 bit transitions
- uniform patterns are kept in dedicated bins, and all others are pooled into one bin

#### Workflow in Code

1. Convert image to grayscale.
2. Compare neighbors to the center pixel.
3. Build 8-bit LBP codes.
4. Count transitions.
5. Map codes into 59 bins.
6. Build histogram.
7. L1-normalize.

#### Theory Check

Mostly aligned with standard uniform LBP.

#### Implementation Check

The code now pads the grayscale image with **reflected borders** before comparing neighbors.

That means border pixels now compare to local reflected content instead of wrapping to the opposite side of the image. This is much closer to the intended local-neighborhood idea behind LBP.

---

### `extract_hog_pooled_torch(...)`

#### Goal

Compute a compact, pooled HOG descriptor.

#### Theory

HOG works by:

1. computing local gradients
2. accumulating gradient orientations into cell histograms
3. normalizing over blocks
4. using the resulting local orientation structure as a descriptor

This is one of the most established gradient-based descriptors.

#### Workflow in Code

1. Convert image to grayscale.
2. Resize.
3. Compute gradients with Sobel.
4. Convert to magnitude and angle.
5. Build per-cell orientation histograms.
6. Normalize over overlapping blocks.
7. Spatially pool block descriptors into a small grid.
8. Flatten the pooled result.

#### Theory Check

Yes, this is a valid compact HOG-style descriptor.

#### Caveats

- The implementation uses **hard binning**, whereas classical Dalal-Triggs HOG often uses bilinear voting.
- With the current default `out_grid=(1,1)`, the actual output is:

```text
1 * 1 * (2*2*9) = 36
```

The docstring now explains that correctly and also gives `out_grid=(4,4)` as the 576-feature example.

---

### `extract_glcm_haralick_torch(...)`

#### Goal

Compute a compact GLCM-based texture descriptor.

#### Theory

A Gray-Level Co-occurrence Matrix (GLCM) counts how often intensity pairs occur at specified offsets and directions.

Haralick-style texture analysis then derives statistics from the GLCM, such as:

- contrast
- homogeneity
- energy
- correlation

#### Workflow in Code

1. Convert image to grayscale.
2. Quantize intensities to a small number of levels.
3. Compute GLCMs for multiple distances and angles.
4. Extract six properties with `graycoprops`.
5. Average each property across all distances and angles.

#### Theory Check

Yes, this is a valid GLCM-based texture descriptor.

#### Caveat

Strictly speaking, this is **Haralick-like**, not the full original Haralick 13/14-feature set. It uses 6 common summary statistics, which is perfectly acceptable if the goal is a compact descriptor.

---

### `extract_gabor_features_torch(...)`

#### Goal

Compute texture features from a bank of Gabor filters.

#### Theory

Gabor filters respond to image structures at specific:

- orientations
- frequencies

They are widely used for texture characterization.

#### Workflow in Code

1. Convert image to grayscale.
2. Optionally resize.
3. For each `(theta, lambda)` pair:
   - build a Gabor kernel
   - filter the image
   - compute mean and standard deviation of the response
4. Concatenate the response statistics.

#### Theory Check

Yes, this is a standard compact Gabor filter-bank descriptor.

#### Caveat

This uses only first-order summary statistics of the response field. That is a compact and valid choice, but not the richest possible Gabor representation.

---

### `extract_edge_histogram_torch(...)`

#### Goal

Compute a compact edge orientation histogram over spatial blocks.

#### Theory

An edge histogram describes how edge directions are distributed spatially across the image. This is related in spirit to MPEG-style edge histograms, though there are many variants.

#### Workflow in Code

1. Convert image to grayscale.
2. Resize.
3. Detect edges with Canny.
4. Compute gradient magnitude and orientation with Sobel.
5. Divide image into spatial blocks.
6. For each edge pixel, assign one of 5 direction bins:
   - horizontal
   - vertical
   - diagonal `+45`
   - diagonal `-45`
   - nondirectional
7. Concatenate histograms and normalize.

#### Theory Check

This is a reasonable custom edge-histogram descriptor.

#### Caveat

This is **not an exact MPEG-7 edge histogram implementation**. It is a simpler custom version based on nearest orientation bins and a magnitude threshold for “nondirectional” edges.

---

## Unified Extraction

### `extract_all_features_torch(...)`

#### Goal

Concatenate all selected descriptors into one feature vector for shallow learning.

#### Workflow

1. Build a list of descriptor extractors.
2. Optionally append SIFT BoVW and ORB BoVW if centers are available.
3. Iterate through extractors.
4. Concatenate all resulting tensors with `torch.cat(...)`.

The current order in the feature vector is:

1. Hu moments
2. Zernike moments
3. Fourier descriptors
4. Affine invariants
5. Log-polar FFT
6. Color histogram
7. Color moments
8. LBP
9. HOG
10. GLCM Haralick
11. Gabor
12. Edge histogram
13. Optional SIFT BoVW
14. Optional ORB BoVW

#### Theory Check

The concatenation idea is standard for handcrafted-feature pipelines.

#### Implementation Check

The optional SIFT/ORB groups are still appended at the end here, and `get_all_feature_names(...)` now mirrors that order.

So the unified extractor is conceptually fine and the name/value bookkeeping is now consistent when BoVW features are turned on.

---

## Final Verdict

Overall, [sl_methods.py](f:/01_Univalle/01_TG/01_Python/sl_methods.py) is a solid handcrafted-feature toolkit. Most implementations are theoretically consistent with the intended descriptors.

### Methods that are well aligned with theory

- Hu moments
- Zernike moments
- Fourier descriptors
- SIFT BoVW
- HSV histogram
- Color moments
- HOG-style pooled descriptor
- GLCM/Haralick-like descriptor
- Gabor filter-bank descriptor

### Methods that are usable but come with important caveats

- **Log-polar FFT**: theory is fine, and the implementation is now stabilized, but it is still worth rechecking on your real dataset after regeneration
- **ORB BoVW**: Hamming matching is now correct, but a binary-aware codebook would still be cleaner than Euclidean k-means prototypes
- **Edge histogram**: valid custom descriptor, but not an exact standard MPEG-7 implementation

### Non-theory implementation issues worth remembering

- none of the earlier bookkeeping mismatches remain in the current implementation, but old CSVs may still contain deprecated legacy columns such as `affine_6` or `f39`

### Practical takeaway

If you are presenting this pipeline in a report or thesis, you can describe it as:

- a **multi-descriptor handcrafted feature framework**
- combining **shape, texture, color, gradient, and frequency information**
- with a few descriptors implemented as **compact approximations** rather than full canonical versions

That would be both accurate and fair to the code.

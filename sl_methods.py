import torch
import numpy as np
import cv2
import math
from tqdm import tqdm


def get_all_feature_names(
    use_sift: bool = True,
    use_orb: bool = True,
    sift_centers: np.ndarray = None,
    orb_centers: np.ndarray = None,
    zernike_degree: int = 6,
    fourier_k: int = 20,
    log_polar_shape: tuple[int, int] = (8, 8),
    color_hist_bins: tuple[int, int, int] = (4, 4, 4),
    hog_block_size: int = 2,
    hog_num_bins: int = 9,
    hog_out_grid: tuple[int, int] = (1, 1),
    gabor_thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    gabor_lambdas=(8.0, 16.0),
    edge_grid: tuple[int, int] = (2, 2),
) -> list[str]:
    """
    Return the feature names in the same order used by extract_all_features_torch.
    """
    names: list[str] = []

    names.extend([f"hu_{i}" for i in range(1, 8)])
    names.extend([f"zernike_{i}" for i in range(1, zernike_degree + 2)])
    names.extend([f"fourier_{i}" for i in range(1, fourier_k + 1)])
    names.extend([f"affine_{i}" for i in range(1, 6)])

    radial_bins, angular_bins = log_polar_shape
    for r in range(radial_bins):
        for a in range(angular_bins):
            names.append(f"logpolarfft_r{r + 1}_a{a + 1}")

    h_bins, s_bins, v_bins = color_hist_bins
    for h in range(h_bins):
        for s in range(s_bins):
            for v in range(v_bins):
                names.append(f"hsvhist_h{h + 1}_s{s + 1}_v{v + 1}")

    color_channels = ("r", "g", "b")
    color_stats = ("mean", "std", "skew")
    for channel in color_channels:
        for stat in color_stats:
            names.append(f"colormoment_{channel}_{stat}")

    names.extend([f"lbp_{i}" for i in range(1, 60)])

    hog_block_dim = hog_block_size * hog_block_size * hog_num_bins
    gy, gx = hog_out_grid
    for y in range(gy):
        for x in range(gx):
            for k in range(hog_block_dim):
                names.append(f"hog_y{y + 1}_x{x + 1}_{k + 1}")

    names.extend([
        "glcm_contrast",
        "glcm_dissimilarity",
        "glcm_homogeneity",
        "glcm_asm",
        "glcm_energy",
        "glcm_correlation",
    ])

    for theta_idx in range(len(gabor_thetas)):
        for lambda_idx in range(len(gabor_lambdas)):
            names.append(f"gabor_theta{theta_idx + 1}_lambda{lambda_idx + 1}_mean")
            names.append(f"gabor_theta{theta_idx + 1}_lambda{lambda_idx + 1}_std")

    edge_orientations = ("horizontal", "vertical", "diag_pos", "diag_neg", "nondirectional")
    edge_gy, edge_gx = edge_grid
    for y in range(edge_gy):
        for x in range(edge_gx):
            for orientation in edge_orientations:
                names.append(f"edgehist_y{y + 1}_x{x + 1}_{orientation}")

    # Keep optional local-descriptor groups at the end to mirror
    # extract_all_features_torch exactly.
    if use_sift and sift_centers is not None:
        names.extend([f"sift_bovw_{i}" for i in range(1, sift_centers.shape[0] + 1)])
    if use_orb and orb_centers is not None:
        names.extend([f"orb_bovw_{i}" for i in range(1, orb_centers.shape[0] + 1)])

    return names


# -------------------------------------------------------------------------
# Helper: tensor <-> numpy conversions
# -------------------------------------------------------------------------

def _to_numpy_gray(img_t: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch RGB tensor (C,H,W) in [0,1] to a grayscale uint8 image
    usable by OpenCV.

    Steps:
    - Ensure tensor is on CPU and detached from graph.
    - Clamp to [0,1] and convert to numpy array (H,W,C).
    - Convert RGB -> BGR because OpenCV expects BGR.
    - Convert to grayscale with cvtColor.
    """
    if img_t.dim() != 3:
        raise ValueError("Image must have shape (C,H,W)")
    # (C,H,W) -> (H,W,C), move to CPU, convert to numpy
    img = img_t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    # Convert [0,1] -> [0,255] uint8
    img_u8 = (img * 255).astype(np.uint8)
    # RGB -> BGR for OpenCV
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    # BGR -> Gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray

def _to_numpy_bgr(img_t: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch RGB tensor (C,H,W) in [0,1] to a BGR uint8 image
    for OpenCV color-based operations (histogram, moments, SIFT, ORB).
    """
    img = img_t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_u8 = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    return img_bgr

# -------------------------------------------------------------------------
# 1. Hu Moments (7 features)
# -------------------------------------------------------------------------

def extract_hu_moments_torch(img_t: torch.Tensor) -> torch.Tensor:
    """
    Compute 7 Hu invariant moments from the grayscale image.

    - Uses OpenCV's spatial moments and Hu moments.
    - Applies log transform with sign to keep values in a reasonable range.
    - Returns a 1D tensor of length 7.
    """
    gray = _to_numpy_gray(img_t)
    # Spatial moments
    m = cv2.moments(gray)
    # 7 Hu moments (numpy array shape (7,1))
    hu = cv2.HuMoments(m).flatten()
    # Log transform commonly used in literature
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return torch.from_numpy(hu_log.astype(np.float32))

# -------------------------------------------------------------------------
# 2. Zernike Moments (first N magnitudes)
# -------------------------------------------------------------------------

def extract_zernike_moments_torch(
    img_t: torch.Tensor,
    radius: int = 64,
    degree: int = 6
) -> torch.Tensor:
    """
    Compute Zernike moments of a **single fixed degree n = degree**.

    - Input is a segmented bird image (C,H,W) in [0,1].
    - We convert to grayscale, resize to (2*radius+1)^2,
      and binarize (Otsu) to emphasize the shape.
    - We then compute Z_{n,m} for all valid m such that:
        * m ∈ {-n, -n+2, ..., n-2, n}
        * (n - |m|) is even  (standard Zernike condition)
    - We return ONLY the magnitudes |Z_{n,m}| in a fixed order of m.

    Output
    ------
    features : torch.Tensor of shape (degree + 1,)
        Corresponds to m = -n, -n+2, ..., n-2, n
        (there are n+1 values for a given degree n).
    """
    gray = _to_numpy_gray(img_t)

    # 1) Resize to square grid around the unit disk
    size = 2 * radius + 1
    gray_resized = cv2.resize(gray, (size, size))

    # 2) Binarize: bird vs background (assuming segmented image)
    #    We use 0/1 instead of 0/255 for numeric stability
    _, binary = cv2.threshold(
        gray_resized, 0, 1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = binary.shape
    y, x = np.indices((h, w))
    x = x - radius          # center at (0,0)
    y = radius - y          # flip y-axis to match usual convention

    r = np.sqrt(x**2 + y**2) / radius
    theta = np.arctan2(y, x)

    # Consider only pixels inside the unit disk
    mask = r <= 1.0
    r_masked = r[mask]
    theta_masked = theta[mask]
    f_masked = binary[mask]

    n = int(degree)
    if n < 0:
        raise ValueError("degree must be non-negative")

    def radial_poly_fixed_n_m(n, m, r_vals):
        """
        Compute radial polynomial R_n^m(r) for a fixed (n,m).

        R_n^m(r) = sum_{s=0}^{(n - |m|)/2} (-1)^s * (n-s)! /
                   [ s! ((n+|m|)/2 - s)! ((n-|m|)/2 - s)! ] * r^{n-2s}
        """
        R = np.zeros_like(r_vals, dtype=np.float64)
        m_abs = abs(m)
        # number of terms in the sum
        max_s = (n - m_abs) // 2
        for s in range(max_s + 1):
            c = ((-1) ** s *
                 math.factorial(n - s) /
                 (math.factorial(s) *
                  math.factorial((n + m_abs)//2 - s) *
                  math.factorial((n - m_abs)//2 - s)))
            R += c * (r_vals ** (n - 2*s))
        return R

    features = []
    # m goes from -n to n in steps of 2 (same parity as n)
    for m in range(-n, n+1, 2):
        # Additional safety: enforce Zernike parity condition
        if (n - abs(m)) % 2 != 0:
            # Skip invalid (n,m) combinations
            continue

        R_nm = radial_poly_fixed_n_m(n, m, r_masked)
        # Zernike basis: V_n^m(r,theta) = R_n^m(r) * e^{-j m theta}
        V_nm = R_nm * np.exp(-1j * m * theta_masked)

        # Discrete approximation of the integral:
        # Z_n^m ≈ (n+1)/π * Σ f(x,y) V_n^m(r,θ)
        Z_nm = (f_masked * V_nm).sum() * (n + 1) / np.pi

        features.append(np.abs(Z_nm))   # magnitude only

    # features length should be n+1 (for valid m)
    features = np.array(features, dtype=np.float32)
    return torch.from_numpy(features)

# -------------------------------------------------------------------------
# 3. Fourier Descriptors on contour (K features)
# -------------------------------------------------------------------------

def extract_fourier_descriptors_torch(img_t: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    Compute contour-based Fourier descriptors.

    Steps:
    - Convert to grayscale and detect edges (Canny).
    - Find the largest contour (assumed to correspond to the bird).
    - Represent the contour as complex numbers x + j*y.
    - Apply 1D FFT to this complex sequence.
    - Take the first k magnitudes (excluding the DC component),
      normalized by the first non-zero magnitude for scale invariance.

    Output:
    - 1D tensor of size k (if contour found), else zeros.
    """
    gray = _to_numpy_gray(img_t)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours; if none, return zeros
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return torch.zeros(k, dtype=torch.float32)

    # Take largest contour by length
    contour = max(contours, key=lambda c: c.shape[0])
    contour = contour.squeeze(1)  # (N,2) -> x,y

    # Convert to complex sequence
    z = contour[:, 0] + 1j * contour[:, 1]
    # Remove mean to be translation invariant
    z = z - z.mean()
    # 1D FFT
    F = np.fft.fft(z)
    # Magnitudes
    mag = np.abs(F)

    # Skip DC (index 0) and take next k
    mag = mag[1:k+1]
    # Scale invariance: divide by first magnitude if nonzero
    if mag[0] != 0:
        mag = mag / mag[0]
    # If contour shorter than k, pad with zeros
    if mag.shape[0] < k:
        mag = np.pad(mag, (0, k - mag.shape[0]), mode="constant")

    return torch.from_numpy(mag.astype(np.float32))

# -------------------------------------------------------------------------
# 4. Affine Invariant Moments (5 features)
# -------------------------------------------------------------------------

def extract_affine_invariants_torch(img_t: torch.Tensor) -> torch.Tensor:
    """
    Compute 5 distinct affine invariant combinations of central moments.

    Steps:
    - Convert to grayscale float64.
    - Compute raw moments m_pq and central moments mu_pq.
    - Normalize moments to eta_pq.
    - Build 5 classical affine invariant combinations.

    Output:
    - 1D tensor of 5 floats.
    """
    gray = _to_numpy_gray(img_t).astype(np.float64)
    h, w = gray.shape
    y, x = np.mgrid[0:h, 0:w]

    def raw_moment(p, q):
        return np.sum((x ** p) * (y ** q) * gray)

    m00 = raw_moment(0, 0) + 1e-12  # avoid division by zero
    x_bar = raw_moment(1, 0) / m00
    y_bar = raw_moment(0, 1) / m00

    def central_moment(p, q):
        return np.sum(((x - x_bar) ** p) * ((y - y_bar) ** q) * gray)

    def eta(p, q):
        return central_moment(p, q) / (m00 ** (1 + (p + q) / 2))

    eta20 = eta(2, 0)
    eta02 = eta(0, 2)
    eta11 = eta(1, 1)
    eta30 = eta(3, 0)
    eta12 = eta(1, 2)
    eta21 = eta(2, 1)
    eta03 = eta(0, 3)

    ami = []
    ami.append(eta20 * eta02 - eta11 ** 2)
    ami.append((eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2)
    ami.append((eta30 + eta12) ** 2 + (eta21 + eta03) ** 2)
    ami.append((eta30 - 3 * eta12) * (eta30 + eta12) *
               ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) +
               (3 * eta21 - eta03) * (eta21 + eta03) *
               (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2))
    ami.append((eta20 - eta02) *
               ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) +
               4 * eta11 * (eta30 + eta12) * (eta21 + eta03))

    return torch.tensor(ami, dtype=torch.float32)

# -------------------------------------------------------------------------
# 5. Log-Polar FFT features (K_r x K_t summary)
# -------------------------------------------------------------------------

def extract_log_polar_fft_torch(img_t: torch.Tensor,
                                out_radial: int = 8,
                                out_angular: int = 8) -> torch.Tensor:
    """
    Compute log-polar FFT features as a compact descriptor.

    Steps:
    - Convert to grayscale float image in [0, 1].
    - Remove the mean and apply a Hanning window to reduce boundary artifacts.
    - Compute 2D FFT and shift zero frequency to center.
    - Take log-magnitude spectrum and suppress the DC peak.
    - Robustly normalize the spectrum before the log-polar transform.
    - Apply OpenCV log-polar transform with a scale derived from image size.
    - Replace any non-finite values and L2-normalize the final descriptor.

    Output:
    - 1D tensor of size out_radial * out_angular.
    """
    gray = _to_numpy_gray(img_t).astype(np.float32) / 255.0
    if gray.size == 0 or gray.shape[0] < 2 or gray.shape[1] < 2:
        return torch.zeros(out_radial * out_angular, dtype=torch.float32)

    # Near-constant images do not have a meaningful frequency signature.
    if float(gray.std()) < 1e-8:
        return torch.zeros(out_radial * out_angular, dtype=torch.float32)

    h, w = gray.shape
    gray = gray - gray.mean()
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    gray = gray * window

    # 2D FFT using numpy (could also use torch.fft)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))  # log for dynamic range
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)

    center = (w // 2, h // 2)
    magnitude[center[1], center[0]] = 0.0  # suppress dominant DC peak

    # Normalize using a robust high percentile instead of the absolute max,
    # which reduces the effect of isolated extreme bins.
    scale = float(np.percentile(magnitude, 99.5))
    if not np.isfinite(scale) or scale <= 0.0:
        return torch.zeros(out_radial * out_angular, dtype=torch.float32)
    magnitude = np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)

    # Log-polar transform (M, N) where M ~ radial, N ~ angular
    # Use a radius-aware scale instead of a fixed heuristic so the mapping
    # behaves consistently across image sizes.
    max_radius = float(min(center[0], center[1]))
    if max_radius <= 1.0:
        return torch.zeros(out_radial * out_angular, dtype=torch.float32)
    M = max(h, w) / np.log(max_radius + 1.0)
    log_polar = cv2.logPolar(
        magnitude,
        center,
        M,
        cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
    )
    log_polar = np.nan_to_num(log_polar, nan=0.0, posinf=0.0, neginf=0.0)
    log_polar = np.clip(log_polar, 0.0, 1.0)

    # Resize to desired resolution and flatten
    log_polar_resized = cv2.resize(
        log_polar,
        (out_angular, out_radial),
        interpolation=cv2.INTER_AREA,
    )
    feat = np.nan_to_num(log_polar_resized.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(feat))
    if norm > 0.0:
        feat = feat / norm
    return torch.from_numpy(feat.astype(np.float32))

# -------------------------------------------------------------------------
# 6. SIFT descriptors -> Bag-of-Visual-Words (fixed length)
# -------------------------------------------------------------------------

def extract_sift_bovw_torch(img_t: torch.Tensor,
                            kmeans_centers: np.ndarray) -> torch.Tensor:
    """
    Compute a fixed-length SIFT Bag-of-Visual-Words descriptor.

    Assumes:
    - `kmeans_centers` is a numpy array of shape (K, 128) obtained by
      running k-means on SIFT descriptors over the training set.

    Steps:
    - Detect SIFT keypoints + descriptors (N x 128).
    - For each descriptor, find nearest visual word (argmin L2).
    - Build histogram of visual words (length K).
    - L1-normalize histogram.

    Output:
    - 1D tensor of length K.
    """
    gray = _to_numpy_gray(img_t)
    sift = cv2.SIFT_create()
    _, desc = sift.detectAndCompute(gray, None)

    K = kmeans_centers.shape[0]
    if desc is None or desc.shape[0] == 0:
        # No keypoints found: return zeros
        return torch.zeros(K, dtype=torch.float32)

    # Compute squared Euclidean distance to all centers
    # desc: (N,128), centers: (K,128)
    # -> distances (N,K)
    diff = desc[:, None, :] - kmeans_centers[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
    # Nearest center index for each descriptor
    words = np.argmin(dist2, axis=1)

    # Histogram over visual words
    hist, _ = np.histogram(words, bins=np.arange(K+1))
    hist = hist.astype(np.float32)
    # L1 normalize
    if hist.sum() > 0:
        hist /= hist.sum()

    return torch.from_numpy(hist)

# -------------------------------------------------------------------------
# 7. ORB descriptors -> Bag-of-Visual-Words (fixed length)
# -------------------------------------------------------------------------

def extract_orb_bovw_torch(img_t: torch.Tensor,
                           kmeans_centers: np.ndarray) -> torch.Tensor:
    """
    Compute a fixed-length ORB Bag-of-Visual-Words descriptor.

    Similar to SIFT BoVW but for ORB (32-byte / 256-bit binary descriptors).

    Assumes:
    - `kmeans_centers` has shape (K, 32), one prototype per visual word.
    - Assignment is done with Hamming distance, which is the natural metric for ORB.

    Note:
    - If the provided centers come from Euclidean k-means, we round them back to
      uint8 byte patterns before Hamming assignment. That keeps matching aligned
      with ORB's binary-descriptor theory, although a binary-aware codebook would
      still be the cleanest training-time choice.

    Output:
    - 1D tensor of length K.
    """
    gray = _to_numpy_gray(img_t)
    orb = cv2.ORB_create()
    _, desc = orb.detectAndCompute(gray, None)

    K = kmeans_centers.shape[0]
    if desc is None or desc.shape[0] == 0:
        return torch.zeros(K, dtype=torch.float32)

    centers_u8 = np.asarray(kmeans_centers)
    if centers_u8.ndim != 2 or centers_u8.shape[1] != 32:
        raise ValueError(
            "ORB BoVW centers must have shape (K, 32) because ORB returns 32-byte descriptors."
        )
    if centers_u8.dtype != np.uint8:
        centers_u8 = np.clip(np.rint(centers_u8), 0, 255).astype(np.uint8)

    desc_u8 = desc.astype(np.uint8, copy=False)
    xor = np.bitwise_xor(desc_u8[:, None, :], centers_u8[None, :, :])
    hamming_dist = np.unpackbits(xor, axis=2).sum(axis=2)
    words = np.argmin(hamming_dist, axis=1)

    hist, _ = np.histogram(words, bins=np.arange(K+1))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()

    return torch.from_numpy(hist)

# -------------------------------------------------------------------------
# 8. Color Histogram in HSV (bins per channel)
# -------------------------------------------------------------------------

def extract_color_histogram_torch(img_t: torch.Tensor,
                                  bins=(4, 4, 4)) -> torch.Tensor:
    """
    Compute a 3D color histogram in HSV color space.

    Steps:
    - Convert RGB tensor to BGR uint8 for OpenCV.
    - Convert BGR -> HSV.
    - Compute 3D histogram over H,S,V with given number of bins.
    - Normalize the histogram so sum = 1.
    - Flatten to 1D vector.

    Output:
    - 1D tensor of length bins[0] * bins[1] * bins[2].
    """
    img_bgr = _to_numpy_bgr(img_t)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],
                        channels=[0, 1, 2],
                        mask=None,
                        histSize=bins,
                        ranges=[0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32)
    # L1 normalize
    if hist.sum() > 0:
        hist /= hist.sum()
    return torch.from_numpy(hist.flatten())

# -------------------------------------------------------------------------
# 9. Color Moments (mean, std, skew per channel)
# -------------------------------------------------------------------------

def extract_color_moments_torch(img_t: torch.Tensor) -> torch.Tensor:
    """
    Compute 3 moments (mean, standard deviation, skewness) for each
    of the 3 color channels (RGB).

    Steps:
    - Use the input tensor (C,H,W) in [0,1].
    - For each channel:
      * mean = average intensity
      * std  = standard deviation
      * skew = E[(x - mean)^3] / std^3

    Output:
    - 1D tensor of length 9.
    """
    if img_t.dim() != 3 or img_t.shape[0] != 3:
        raise ValueError("Image must have shape (3,H,W)")

    feats = []
    for c in range(3):
        chan = img_t[c]  # (H,W)
        mean = chan.mean()
        std = chan.std(unbiased=False) + 1e-8
        skew = torch.mean(((chan - mean) ** 3)) / (std ** 3)
        feats.extend([mean, std, skew])

    return torch.stack(feats).float()

# -------------------------------------------------------------------------
# 10. Local Binary Patterns
# -------------------------------------------------------------------------

def extract_lbp_uniform_hist_torch(
    img_t: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute a Uniform LBP (Local Binary Pattern) histogram.

    - Input: img_t (C,H,W) in [0,1]
    - Uses standard LBP with P=8, R=1
    - Keeps only 'uniform' patterns (<= 2 bit transitions)
    - Output size: 59 features (58 uniform + 1 non-uniform bin)

    Border handling uses reflected padding so border pixels compare against
    local neighbors instead of wrapping around to the opposite side.

    This is the most common compact LBP descriptor in literature.
    """
    gray = _to_numpy_gray(img_t).astype(np.uint8)

    H, W = gray.shape
    lbp = np.zeros((H, W), dtype=np.uint8)
    gray_padded = np.pad(gray, pad_width=1, mode="reflect")
    center = gray_padded[1:-1, 1:-1]

    neighbors = [
        (-1,  0), (-1,  1), (0,  1), (1,  1),
        (1,  0), (1, -1), (0, -1), (-1, -1)
    ]

    # Build LBP code
    for i, (dy, dx) in enumerate(neighbors):
        shifted = gray_padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        lbp |= ((shifted >= center).astype(np.uint8) << i)

    # Count transitions (circular binary string)
    def transitions(code):
        b = np.binary_repr(code, width=8)
        return sum(b[i] != b[(i + 1) % 8] for i in range(8))

    # Build mapping: uniform patterns → [0..57], non-uniform → 58
    mapping = np.zeros(256, dtype=np.uint8)
    idx = 0
    for i in range(256):
        if transitions(i) <= 2:
            mapping[i] = idx
            idx += 1
        else:
            mapping[i] = 58  # non-uniform bin

    lbp_mapped = mapping[lbp]

    hist = np.bincount(lbp_mapped.ravel(), minlength=59).astype(np.float32)

    if normalize and hist.sum() > 0:
        hist /= hist.sum()

    return torch.from_numpy(hist)

# -------------------------------------------------------------------------
# 11. HOG function
# -------------------------------------------------------------------------
 
def extract_hog_pooled_torch(
    img_t: torch.Tensor,
    cell_size: int = 8,
    num_bins: int = 9,
    block_size: int = 2,                 # 2x2 cells per block (classic)
    resize_to: tuple[int, int] = (128, 128),  # smaller = faster
    out_grid: tuple[int, int] = (1, 1),  # pooling grid: (Gy, Gx)
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute a compact HOG descriptor with block normalization + spatial pooling.

    Why pooling?
    - "Full" HOG with 224x224, cell=8, block=2 produces a huge vector (~26k).
    - Here we compute block-normalized HOG and then average-pool it into a small grid
      to keep features conservative and fast.

    Output size:
      out_grid_y * out_grid_x * (block_size*block_size*num_bins)
    Current default: 1*1*(2*2*9) = 36 features.

    Example:
      if out_grid=(4, 4), then 4*4*(2*2*9) = 576 features.

    Workflow:
    1) Convert tensor -> grayscale uint8 via _to_numpy_gray(img_t)
    2) Resize to resize_to for speed/stability
    3) Compute gradients (Sobel), magnitude + angle (0..180)
    4) Build per-cell orientation histograms (num_bins)
    5) Build per-block normalized vectors (2x2 cells -> 36 dims)
    6) Average-pool blocks into out_grid and flatten
    """
    gray = _to_numpy_gray(img_t)  # uint8 (H,W)
    gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)

    H, W = gray.shape

    # ---- 1) Gradients ----
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)  # 0..360
    ang = np.mod(ang, 180.0)                       # unsigned -> 0..180

    # ---- 2) Cell histograms ----
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size
    if n_cells_y < block_size or n_cells_x < block_size:
        # not enough cells to form even one block
        feat_dim = out_grid[0] * out_grid[1] * (block_size * block_size * num_bins)
        return torch.zeros(feat_dim, dtype=torch.float32)

    # crop to exact multiple of cell_size
    Hc = n_cells_y * cell_size
    Wc = n_cells_x * cell_size
    mag = mag[:Hc, :Wc]
    ang = ang[:Hc, :Wc]

    cell_hist = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)
    bin_width = 180.0 / num_bins

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * cell_size
            x0 = cx * cell_size
            m_patch = mag[y0:y0 + cell_size, x0:x0 + cell_size].ravel()
            a_patch = ang[y0:y0 + cell_size, x0:x0 + cell_size].ravel()

            # soft binning is possible, but hard binning is faster/simpler
            bins = np.floor(a_patch / bin_width).astype(np.int32)
            bins = np.clip(bins, 0, num_bins - 1)

            hist = np.zeros(num_bins, dtype=np.float32)
            np.add.at(hist, bins, m_patch)  # accumulate magnitudes into bins
            cell_hist[cy, cx, :] = hist

    # ---- 3) Block normalization (2x2 cells) ----
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    block_dim = block_size * block_size * num_bins  # e.g., 2*2*9 = 36

    blocks = np.zeros((n_blocks_y, n_blocks_x, block_dim), dtype=np.float32)

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cell_hist[by:by + block_size, bx:bx + block_size, :].reshape(-1)
            # L2 normalization
            norm = np.sqrt(np.sum(block * block) + eps * eps)
            blocks[by, bx, :] = block / norm

    # ---- 4) Spatial pooling to out_grid ----
    Gy, Gx = out_grid
    # If blocks are smaller than grid, fall back to global average
    if n_blocks_y < Gy or n_blocks_x < Gx:
        pooled = blocks.mean(axis=(0, 1), keepdims=True)  # (1,1,dim)
        pooled = np.repeat(np.repeat(pooled, Gy, axis=0), Gx, axis=1)
    else:
        pooled = np.zeros((Gy, Gx, block_dim), dtype=np.float32)
        y_edges = np.linspace(0, n_blocks_y, Gy + 1).astype(int)
        x_edges = np.linspace(0, n_blocks_x, Gx + 1).astype(int)

        for gy_i in range(Gy):
            y0, y1 = y_edges[gy_i], y_edges[gy_i + 1]
            for gx_i in range(Gx):
                x0, x1 = x_edges[gx_i], x_edges[gx_i + 1]
                region = blocks[y0:y1, x0:x1, :]
                pooled[gy_i, gx_i, :] = region.mean(axis=(0, 1))

    feat = pooled.reshape(-1).astype(np.float32)
    return torch.from_numpy(feat)

# -------------------------------------------------------------------------
# 12. Haralick/GLCM function
# -------------------------------------------------------------------------
 
def extract_glcm_haralick_torch(
    img_t: torch.Tensor,
    distances=(1, 2),
    angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    levels: int = 32,
    symmetric: bool = True,
    normed: bool = True
) -> torch.Tensor:
    """
    Compute a compact Haralick/GLCM texture descriptor.

    Workflow (matches your other functions):
    - Input: img_t (C,H,W) in [0,1]
    - Convert to grayscale uint8 via _to_numpy_gray(img_t)
    - Quantize grayscale to `levels` (e.g. 32) to keep GLCM cheap
    - Compute GLCM for given distances and angles
    - Extract standard properties and average over all (d, angle)

    Output:
    - 1D tensor of length 6:
        [contrast, dissimilarity, homogeneity, ASM, energy, correlation]
    """
    # Lazy import so the rest of your pipeline works even if skimage is missing
    from skimage.feature import graycomatrix, graycoprops

    gray = _to_numpy_gray(img_t)  # uint8 (H,W)

    # ---- 1) Quantize intensities to reduce compute ----
    # Map 0..255 -> 0..levels-1
    if levels < 2 or levels > 256:
        raise ValueError("levels must be in [2, 256]")
    gray_q = (gray.astype(np.float32) * (levels - 1) / 255.0).astype(np.uint8)

    # ---- 2) Compute GLCM ----
    glcm = graycomatrix(
        gray_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=symmetric,
        normed=normed
    )
    # glcm shape: (levels, levels, len(distances), len(angles))

    # ---- 3) Extract Haralick-like properties ----
    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    feats = []
    for p in props:
        val = graycoprops(glcm, p)  # shape: (len(distances), len(angles))
        feats.append(val.mean())    # average across distances and angles

    return torch.tensor(feats, dtype=torch.float32)

# -------------------------------------------------------------------------
# 13. Gabor filter
# -------------------------------------------------------------------------

def extract_gabor_features_torch(
    img_t: torch.Tensor,
    resize_to: tuple[int, int] = (128, 128),
    thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    lambdas=(8.0, 16.0),
    sigma: float = 4.0,
    gamma: float = 0.5,
    psi: float = 0.0
) -> torch.Tensor:
    """
    Compute a compact Gabor filter bank descriptor.

    Workflow:
    - Input: img_t (C,H,W) in [0,1]
    - Convert to grayscale uint8 via _to_numpy_gray(img_t)
    - Resize (optional) to speed up filtering
    - For each (theta, lambda):
        * build a Gabor kernel
        * filter image (cv2.filter2D)
        * compute mean and std of the response magnitude
    - Output: 1D torch tensor

    Output size:
      len(thetas) * len(lambdas) * 2
      Default: 4 * 2 * 2 = 16 features
    """
    gray = _to_numpy_gray(img_t).astype(np.float32)  # (H,W) float
    if resize_to is not None:
        gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)

    feats = []

    for theta in thetas:
        for lamb in lambdas:
            # Kernel size: choose odd size ~ 6*sigma (common heuristic)
            ksize = int(max(7, (6 * sigma) // 1))
            if ksize % 2 == 0:
                ksize += 1

            kernel = cv2.getGaborKernel(
                ksize=(ksize, ksize),
                sigma=sigma,
                theta=theta,
                lambd=lamb,
                gamma=gamma,
                psi=psi,
                ktype=cv2.CV_32F
            )

            # Filter response
            resp = cv2.filter2D(gray, cv2.CV_32F, kernel)

            # Use magnitude statistics (mean, std)
            mean_val = float(resp.mean())
            std_val = float(resp.std())
            feats.extend([mean_val, std_val])

    return torch.tensor(feats, dtype=torch.float32)

# -------------------------------------------------------------------------
# 14. Edge histogram
# -------------------------------------------------------------------------

def extract_edge_histogram_torch(
    img_t: torch.Tensor,
    resize_to: tuple[int, int] = (128, 128),
    grid: tuple[int, int] = (2, 2),
    canny1: int = 80,
    canny2: int = 160,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute a compact edge histogram descriptor.

    Workflow:
    - Input: img_t (C,H,W) in [0,1]
    - Convert to grayscale uint8 via _to_numpy_gray(img_t)
    - Resize to resize_to for speed/stability
    - Run Canny to get edge mask
    - Compute gradient orientation via Sobel on grayscale
    - Divide image into grid blocks (default 4x4)
    - For each block, count edge pixels falling into 5 orientation bins:
        0) horizontal
        1) vertical
        2) diag +45
        3) diag -45
        4) non-directional / weak-orientation
    - Concatenate and L1-normalize

    Output size:
      grid_y * grid_x * 5
      Default: 4*4*5 = 80 features
    """
    gray = _to_numpy_gray(img_t)  # uint8
    if resize_to is not None:
        gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)

    # Canny edges (binary mask)
    edges = cv2.Canny(gray, canny1, canny2)

    # Sobel gradients for orientation
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)  # 0..360
    ang = np.mod(ang, 180.0)  # unsigned -> 0..180

    H, W = gray.shape
    Gy, Gx = grid
    bh = H // Gy
    bw = W // Gx

    feats = []

    # Orientation binning thresholds (degrees)
    # Define simple bins around 0, 45, 90, 135.
    # We'll classify based on nearest of these directions; weak magnitude -> non-directional.
    weak_thr = 10.0  # gradient magnitude threshold for "non-directional"

    def orient_bin(angle_deg: float, m: float) -> int:
        if m < weak_thr:
            return 4  # non-directional
        # nearest among 0,45,90,135
        candidates = np.array([0.0, 45.0, 90.0, 135.0], dtype=np.float32)
        idx = int(np.argmin(np.abs(candidates - angle_deg)))
        return idx  # 0..3

    for gy_i in range(Gy):
        for gx_i in range(Gx):
            y0 = gy_i * bh
            x0 = gx_i * bw
            y1 = (gy_i + 1) * bh if gy_i < Gy - 1 else H
            x1 = (gx_i + 1) * bw if gx_i < Gx - 1 else W

            e_block = edges[y0:y1, x0:x1]
            a_block = ang[y0:y1, x0:x1]
            m_block = mag[y0:y1, x0:x1]

            # Only consider pixels that are edges
            ys, xs = np.where(e_block > 0)
            hist = np.zeros(5, dtype=np.float32)

            for yy, xx in zip(ys, xs):
                b = orient_bin(float(a_block[yy, xx]), float(m_block[yy, xx]))
                hist[b] += 1.0

            feats.extend(hist.tolist())

    feats = np.array(feats, dtype=np.float32)

    # L1 normalize
    s = feats.sum()
    if s > 0:
        feats /= (s + eps)

    return torch.from_numpy(feats)
    
# -------------------------------------------------------------------------
# 15. Unified feature extractor (concatenate everything you want)
# -------------------------------------------------------------------------

def extract_all_features_torch(
    img_t: torch.Tensor,
    sift_centers: np.ndarray = None,
    orb_centers: np.ndarray = None,
    use_sift: bool = True,
    use_orb: bool = True,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    High-level function that concatenates all selected descriptors into a
    single 1D feature vector for shallow learning.

    You can toggle SIFT/ORB (because they require trained BoVW centers).

    Output:
    - 1D tensor with all concatenated features.
    """
    feats = []
    extractors = [
        ("Hu moments", lambda: extract_hu_moments_torch(img_t)),
        ("Zernike moments", lambda: extract_zernike_moments_torch(img_t)),
        ("Fourier descriptors", lambda: extract_fourier_descriptors_torch(img_t)),
        ("Affine invariants", lambda: extract_affine_invariants_torch(img_t)),
        ("Log-polar FFT", lambda: extract_log_polar_fft_torch(img_t)),
        ("Color histogram", lambda: extract_color_histogram_torch(img_t)),
        ("Color moments", lambda: extract_color_moments_torch(img_t)),
        ("LBP", lambda: extract_lbp_uniform_hist_torch(img_t)),
        ("HOG", lambda: extract_hog_pooled_torch(img_t)),
        ("GLCM Haralick", lambda: extract_glcm_haralick_torch(img_t)),
        ("Gabor", lambda: extract_gabor_features_torch(img_t)),
        ("Edge histogram", lambda: extract_edge_histogram_torch(img_t)),
    ]

    # Local keypoints (if centers provided)
    if use_sift and sift_centers is not None:
        extractors.append(("SIFT BoVW", lambda: extract_sift_bovw_torch(img_t, sift_centers)))
    if use_orb and orb_centers is not None:
        extractors.append(("ORB BoVW", lambda: extract_orb_bovw_torch(img_t, orb_centers)))

    extractor_iter = tqdm(
        extractors,
        desc="Extracting feature groups",
        unit="feature_group",
        leave=False,
        disable=not show_progress,
    )
    for _, extractor in extractor_iter:
        feats.append(extractor())

    # Concatenate all to one long vector
    feature_vector = torch.cat(feats, dim=0)
    return torch.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)



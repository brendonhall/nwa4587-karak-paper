"""Pyroxene exsolution lamellae analysis.

Verbatim port of the published hires analysis (lamellae_from_hires.py): grain
segmentation, structure tensor orientation, autocorrelation spacing and a
Rayleigh test. Settings arrive as arguments; literals became LamellaeSettings
fields. The arithmetic is the old code.
"""

from __future__ import annotations

import numpy as np

from . import constants as C

AUGITE_LABEL = 1      # internal binary label for augite
PIGEONITE_LABEL = 0   # internal binary label for pigeonite


def binary_map(hires: np.ndarray, augite: int, pigeonite: int) -> np.ndarray:
    """int8 map: augite 1, pigeonite 0, everything else -1."""
    label_1x = np.full(hires.shape, -1, dtype=np.int8)
    label_1x[hires == augite] = AUGITE_LABEL
    label_1x[hires == pigeonite] = PIGEONITE_LABEL
    return label_1x


def segment_grains(label_1x: np.ndarray, min_size: int, settings: C.LamellaeSettings
                   ) -> tuple[np.ndarray, int]:
    """Segment pyroxene grains via connected components on combined mask.

    Returns (grain_labels, n_grains).
    """
    from scipy.ndimage import label as ndlabel
    from scipy.ndimage import binary_closing
    from skimage.morphology import remove_small_objects

    pyx_mask = label_1x >= 0  # augite (1) or pigeonite (0)

    # Close small gaps from unresolved boundary pixels
    struct = np.ones((3, 3), dtype=bool)
    pyx_closed = binary_closing(pyx_mask, structure=struct,
                                iterations=settings.closing_iterations)
    del pyx_mask

    # Connected components (8-connectivity)
    grain_labels, n_raw = ndlabel(pyx_closed, structure=struct)
    del pyx_closed

    # Remove small objects (max_size = min_size - 1 keeps objects of size >= min_size)
    grain_labels = remove_small_objects(grain_labels, max_size=min_size - 1,
                                        connectivity=2)
    # Relabel sequentially
    unique = np.unique(grain_labels)
    unique = unique[unique > 0]
    n_grains = len(unique)

    return grain_labels, n_grains


def measure_orientation(augite_binary: np.ndarray, grain_mask: np.ndarray, sigma: float,
                        settings: C.LamellaeSettings
                        ) -> tuple[float, float]:
    """Measure dominant lamellae orientation via structure tensor.

    Returns (orientation_deg, coherence) where orientation is in [0, 180).
    """
    from skimage.feature import structure_tensor, structure_tensor_eigenvalues
    from scipy.ndimage import binary_erosion

    # Erode grain mask to exclude boundary gradient artifacts
    eroded = binary_erosion(grain_mask, iterations=settings.erosion_iterations)
    if eroded.sum() < settings.erosion_fallback_px:
        eroded = grain_mask  # fall back if erosion removes too much

    # Structure tensor on augite binary within grain
    img = augite_binary.astype(np.float64)
    S = structure_tensor(img, sigma=sigma)
    eigenvalues = structure_tensor_eigenvalues(S)

    # eigenvalues shape: (2, H, W), eigenvalues[0] >= eigenvalues[1]
    lam_max = eigenvalues[0]
    lam_min = eigenvalues[1]

    # Coherence per pixel
    denom = lam_max + lam_min
    coherence_px = np.where(denom > 0, (lam_max - lam_min) / denom, 0.0)

    # Orientation per pixel (direction of gradient = perpendicular to lamellae)
    # S components: Axx=S[0], Axy=S[1], Ayy=S[2] (for 2D)
    Axx, Axy, Ayy = S[0], S[1], S[2]
    # Gradient direction
    theta_grad = 0.5 * np.arctan2(2 * Axy, Axx - Ayy)
    # Lamellae direction is perpendicular to gradient
    theta_lam = theta_grad + np.pi / 2

    # Weighted circular mean within eroded grain mask (doubled-angle for axial data)
    mask = eroded & (coherence_px > settings.coherence_px_min)
    if mask.sum() < settings.min_orientation_px:
        return np.nan, 0.0

    weights = coherence_px[mask]
    angles = theta_lam[mask]

    # Doubled-angle method for axial data
    sin_2a = np.sum(weights * np.sin(2 * angles))
    cos_2a = np.sum(weights * np.cos(2 * angles))
    mean_angle = 0.5 * np.arctan2(sin_2a, cos_2a)

    # Convert to [0, 180) degrees
    mean_deg = np.degrees(mean_angle) % 180

    # Mean coherence (weighted)
    mean_coherence = float(np.average(coherence_px[mask], weights=weights))

    return mean_deg, mean_coherence


def measure_spacing(augite_binary: np.ndarray, grain_mask: np.ndarray,
                    orientation_deg: float, n_profiles: int,
                    settings: C.LamellaeSettings
                    ) -> tuple[float, float, int]:
    """Measure lamellae spacing via 1D profile autocorrelation.

    Returns (median_spacing_px, iqr_spacing_px, n_valid_profiles).
    """
    from scipy.ndimage import map_coordinates
    from scipy.signal import find_peaks

    # Direction perpendicular to lamellae (for profile sampling)
    perp_rad = np.radians(orientation_deg + 90)
    dx = np.cos(perp_rad)
    dy = np.sin(perp_rad)

    # Along-lamellae direction (for distributing profiles)
    along_rad = np.radians(orientation_deg)
    ax = np.cos(along_rad)
    ay = np.sin(along_rad)

    # Find grain extent
    rows, cols = np.where(grain_mask)
    if len(rows) == 0:
        return np.nan, np.nan, 0
    cy, cx = rows.mean(), cols.mean()

    # Project grain pixels onto perpendicular and along-lamellae axes
    dr = rows - cy
    dc = cols - cx
    proj_perp = dr * dy + dc * dx
    proj_along = dr * ay + dc * ax

    # Profile extent along perpendicular direction
    perp_min, perp_max = proj_perp.min(), proj_perp.max()
    perp_extent = perp_max - perp_min
    if perp_extent < settings.min_perp_extent_px:  # need at least a few pixels
        return np.nan, np.nan, 0

    # Distribute profiles along the along-lamellae axis
    along_min, along_max = proj_along.min(), proj_along.max()
    along_positions = np.linspace(
        along_min + settings.profile_margin_px,
        along_max - settings.profile_margin_px,
        min(n_profiles,
            max(settings.min_profile_cap,
                int(perp_extent / settings.profile_extent_divisor))))

    spacings = []
    for along_pos in along_positions:
        # Profile center in image coordinates
        center_r = cy + along_pos * ay
        center_c = cx + along_pos * ax

        # Sample along perpendicular direction
        t_vals = np.arange(perp_min, perp_max + 1, 1.0)
        sample_r = center_r + t_vals * dy
        sample_c = center_c + t_vals * dx

        # Check bounds
        h, w = augite_binary.shape
        valid = ((sample_r >= 0) & (sample_r < h - 1) &
                 (sample_c >= 0) & (sample_c < w - 1))
        if valid.sum() < settings.min_profile_len_px:
            continue

        coords = np.array([sample_r[valid], sample_c[valid]])

        # Sample grain mask to ensure we stay within the grain
        grain_profile = map_coordinates(grain_mask.astype(np.float64),
                                        coords, order=0)
        in_grain = grain_profile > 0.5
        if in_grain.sum() < settings.min_profile_len_px:
            continue

        # Sample augite binary
        augite_profile = map_coordinates(augite_binary.astype(np.float64),
                                         coords, order=0)
        profile = augite_profile[in_grain]

        if len(profile) < settings.min_profile_len_px or profile.std() < settings.profile_std_min:
            continue

        # Autocorrelation
        centered = profile - profile.mean()
        acf = np.correlate(centered, centered, mode="full")
        acf = acf[len(acf) // 2:]  # positive lags only
        if acf[0] <= 0:
            continue
        acf = acf / acf[0]

        # Find first significant peak
        peaks, props = find_peaks(acf, distance=settings.peak_distance,
                                  height=settings.peak_height,
                                  prominence=settings.peak_prominence)
        if len(peaks) > 0:
            spacings.append(float(peaks[0]))

    if len(spacings) < settings.min_valid_profiles:
        return np.nan, np.nan, len(spacings)

    spacings = np.array(spacings)
    median_sp = float(np.median(spacings))
    q25, q75 = np.percentile(spacings, [25, 75])
    iqr = float(q75 - q25)

    return median_sp, iqr, len(spacings)


def rayleigh_test(angles_deg: np.ndarray) -> tuple[float, float]:
    """Rayleigh test for uniformity on axial data (0-180 degrees).

    Returns (R_bar, p_value).
    """
    angles_rad = np.radians(angles_deg) * 2  # double for axial data
    n = len(angles_rad)
    C = np.sum(np.cos(angles_rad))
    S = np.sum(np.sin(angles_rad))
    R = np.sqrt(C**2 + S**2)
    R_bar = R / n

    # Approximate p-value (Mardia & Jupp 2000)
    z = n * R_bar**2
    p = np.exp(-z) * (1 + (2 * z - z**2) / (4 * n) -
                       (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4) / (288 * n**2))
    p = max(0.0, min(1.0, p))
    return float(R_bar), float(p)


def analyze(hires: np.ndarray, augite: int, pigeonite: int, pixel_size_um: float,
            settings: C.LamellaeSettings = C.LAMELLAE) -> tuple[dict, list[dict]]:
    """Segment, measure and summarise the lamellae of a hires phase map.

    Returns (summary, grain records). With no analyzed grain every float in the
    summary is NaN and orientation.uniform is None.
    """
    from skimage.measure import regionprops
    from scipy.stats import spearmanr

    label_1x = binary_map(hires, augite, pigeonite)
    grain_labels, n_grains = segment_grains(label_1x, settings.min_grain_px, settings)

    props = regionprops(grain_labels)

    # Filter to grains large enough for analysis
    large_grains = [p for p in props if p.area >= settings.min_analysis_px]

    results = []
    for i, prop in enumerate(large_grains):
        r0, c0, r1, c1 = prop.bbox
        grain_id = prop.label

        # Extract grain-local arrays
        grain_mask_local = (grain_labels[r0:r1, c0:c1] == grain_id)
        augite_local = (label_1x[r0:r1, c0:c1] == AUGITE_LABEL) & grain_mask_local

        # Check augite fraction within grain
        aug_frac = augite_local.sum() / max(grain_mask_local.sum(), 1)
        if aug_frac < settings.augite_fraction_min or aug_frac > settings.augite_fraction_max:
            # No lamellae to measure (pure augite or pure pigeonite)
            continue

        # Orientation
        orient_deg, coherence = measure_orientation(
            augite_local, grain_mask_local, settings.tensor_sigma, settings)

        if np.isnan(orient_deg) or coherence < settings.coherence_min:
            continue

        # Spacing
        spacing_px, spacing_iqr_px, n_valid = measure_spacing(
            augite_local, grain_mask_local, orient_deg,
            n_profiles=settings.n_profiles, settings=settings)

        if np.isnan(spacing_px) or n_valid < settings.min_valid_profiles:
            continue

        spacing_um = spacing_px * pixel_size_um
        spacing_iqr_um = spacing_iqr_px * pixel_size_um
        area_um2 = prop.area * pixel_size_um**2
        equiv_diam_um = np.sqrt(4 * area_um2 / np.pi)

        results.append({
            "grain_id": int(grain_id),
            "area_px": int(prop.area),
            "area_um2": round(area_um2, 1),
            "equiv_diameter_um": round(equiv_diam_um, 1),
            "centroid_row": round(prop.centroid[0], 1),
            "centroid_col": round(prop.centroid[1], 1),
            "bbox": [r0, c0, r1, c1],
            "augite_fraction": round(float(aug_frac), 4),
            "orientation_deg": round(orient_deg, 1),
            "coherence": round(coherence, 4),
            "spacing_px": round(spacing_px, 2),
            "spacing_um": round(spacing_um, 1),
            "spacing_iqr_um": round(spacing_iqr_um, 1),
            "n_profiles": n_valid,
            "near_nyquist": spacing_px < settings.nyquist_px,
        })

    nan = float("nan")
    if len(results) == 0:
        summary = {
            "n_grains_total": n_grains,
            "n_grains_analyzed": 0,
            "pixel_size_um": pixel_size_um,
            "nyquist_limit_um": 2 * pixel_size_um,
            "orientation": {"circular_mean_deg": nan, "rayleigh_R_bar": nan,
                            "rayleigh_p_value": nan, "uniform": None},
            "spacing": {"mean_um": nan, "median_um": nan, "std_um": nan, "q25_um": nan,
                        "q75_um": nan, "min_um": nan, "max_um": nan, "n_near_nyquist": 0},
            "grain_size_vs_spacing": {"spearman_rho": nan, "spearman_p": nan},
            "coherence": {"mean": nan, "min": nan, "max": nan},
        }
        return summary, results

    orientations = np.array([r["orientation_deg"] for r in results])
    spacings_um = np.array([r["spacing_um"] for r in results])
    areas_um2 = np.array([r["area_um2"] for r in results])
    coherences = np.array([r["coherence"] for r in results])
    n_near_nyquist = sum(1 for r in results if r["near_nyquist"])

    # Circular statistics for orientation
    R_bar, rayleigh_p = rayleigh_test(orientations)

    # Circular mean of orientations (axial)
    angles_2x = np.radians(orientations) * 2
    circ_mean_deg = float(np.degrees(
        0.5 * np.arctan2(np.sin(angles_2x).mean(), np.cos(angles_2x).mean())
    ) % 180)

    # Spacing statistics
    sp_corr, sp_p = spearmanr(areas_um2, spacings_um)

    summary = {
        "n_grains_total": n_grains,
        "n_grains_analyzed": len(results),
        "pixel_size_um": pixel_size_um,
        "nyquist_limit_um": 2 * pixel_size_um,
        "orientation": {
            "circular_mean_deg": round(circ_mean_deg, 1),
            "rayleigh_R_bar": round(R_bar, 4),
            "rayleigh_p_value": round(rayleigh_p, 6),
            "uniform": rayleigh_p > 0.05,
        },
        "spacing": {
            "mean_um": round(float(spacings_um.mean()), 1),
            "median_um": round(float(np.median(spacings_um)), 1),
            "std_um": round(float(spacings_um.std()), 1),
            "q25_um": round(float(np.percentile(spacings_um, 25)), 1),
            "q75_um": round(float(np.percentile(spacings_um, 75)), 1),
            "min_um": round(float(spacings_um.min()), 1),
            "max_um": round(float(spacings_um.max()), 1),
            "n_near_nyquist": n_near_nyquist,
        },
        "grain_size_vs_spacing": {
            "spearman_rho": round(float(sp_corr), 4),
            "spearman_p": round(float(sp_p), 6),
        },
        "coherence": {
            "mean": round(float(coherences.mean()), 4),
            "min": round(float(coherences.min()), 4),
            "max": round(float(coherences.max()), 4),
        },
    }
    return summary, results

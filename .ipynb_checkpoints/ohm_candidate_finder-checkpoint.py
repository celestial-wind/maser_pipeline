# ohm_candidate_finder.py

"""
A module for post-processing search outputs (SNR cubes) to identify, 
characterize, and validate candidate signals.

This module provides three main categories of tools:
1.  Candidate Finding Algorithms:
    - A simple 2D peak finder for max-SNR maps.
    - A 3D DBSCAN clustering algorithm for finding candidates in full SNR cubes.

2.  Performance Assessment:
    - Functions to match found candidates against a ground truth list of
      injected signals to calculate search completeness and purity.

3.  Visualization Suite:
    - A set of plotting functions to visually inspect SNR maps, candidate
      locations, signal spectra, and overall search performance.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit
from typing import List, Dict, Any, Tuple, Optional
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import cmocean

import ohm_search_simulator as oss
import ohm_template_generator as otg

# =============================================================================
# --- Candidate Finding Algorithms ---
# =============================================================================


def validate_from_ground_truth(
    snr_cube: np.ndarray,
    ground_truth: dict,
    freqs_mhz: np.ndarray,
    N_pix_x: int,
) -> list:
    """
    Checks the SNR cube at the known locations of injected masers from the ground truth.

    This function bypasses searching and instead directly measures the performance
    of the matched filter on the signals that were known to be injected.

    Args:
        snr_cube: The 3D cube of Signal-to-Noise Ratios.
        ground_truth: The dictionary containing the list of injected masers.
        freqs_mhz: The array of frequency channels.
        N_pix_x: The width of the simulated sky patch in pixels.

    Returns:
        A list of dictionaries, one for each ground truth maser, containing its
        measured SNR and location information.
    """
    validation_results = []
    
    injections = ground_truth.get('injections', [])
    if not injections:
        print("No injections found in the ground_truth dictionary.")
        return []

    print(f"Validating the SNR for {len(injections)} ground truth injections...")

    for maser_info in tqdm(injections, desc="Checking Ground Truth Injections"):
        pixel_idx = maser_info['pixel_index']
        true_z = maser_info['z']
        true_freq = oss.z_to_freq(true_z)
        
        # Find the frequency channel index closest to the maser's true frequency
        freq_idx = np.argmin(np.abs(freqs_mhz - true_freq))
        
        # Look up the SNR value at that exact pixel and frequency channel
        measured_snr = snr_cube[pixel_idx, freq_idx]
        
        # Calculate the spatial (x, y) coordinates from the pixel index
        y_coord = pixel_idx // N_pix_x
        x_coord = pixel_idx % N_pix_x
        
        validation_results.append({
            'max_snr': measured_snr,
            'pixel_idx': pixel_idx,
            'centroid_x': x_coord,
            'centroid_y': y_coord,
            'centroid_freq_mhz': freqs_mhz[freq_idx], # Using the closest channel freq as the centroid
            'centroid_z': true_z, # Using the exact true redshift
            'is_detected': measured_snr > 5.0 # Example detection threshold
        })
        
    return validation_results


def find_candidates_top_n_per_pixel(
    snr_cube: np.ndarray,
    freqs_mhz: np.ndarray,
    N_pix_x: int,
    top_n: int = 3,
    snr_threshold: float = 5.0,
    centroid_window: int = 2
) -> list:
    """
    Finds top N candidates and calculates a full 3D centroid for each.
    """
    candidate_list = []
    num_pixels, num_freqs = snr_cube.shape

    for i in tqdm(range(num_pixels), desc="Finding Top N Candidates Per Pixel"):
        snr_spectrum = snr_cube[i, :]
        if len(snr_spectrum) > top_n:
            top_indices = np.argpartition(snr_spectrum, -top_n)[-top_n:]
        else:
            top_indices = np.arange(len(snr_spectrum))

        for freq_idx in top_indices:
            snr_value = snr_spectrum[freq_idx]
            if snr_value > snr_threshold:
                start = max(0, freq_idx - centroid_window)
                end = min(num_freqs, freq_idx + centroid_window + 1)
                snrs_in_window = snr_cube[i, start:end]
                freqs_in_window = freqs_mhz[start:end]
                
                snr_sum = np.sum(snrs_in_window)
                if snr_sum > 0:
                    centroid_freq = np.sum(snrs_in_window * freqs_in_window) / snr_sum
                else:
                    centroid_freq = freqs_mhz[freq_idx]
                
                centroid_y = i // N_pix_x
                centroid_x = i % N_pix_x

                candidate_list.append({
                    'max_snr': snr_value,
                    'pixel_idx': i,
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                    'centroid_freq_mhz': centroid_freq,
                    'centroid_z_freq': centroid_freq
                })
                
    print(f"Found {len(candidate_list)} candidates above SNR={snr_threshold}.")
    return sorted(candidate_list, key=lambda x: x['max_snr'], reverse=True)


def find_candidates_3d_dbscan(
    snr_cube: np.ndarray,
    snr_threshold: float,
    eps: int = 3,
    min_samples: int = 5,
    freqs_mhz: np.ndarray = None
) -> List[Dict[str, Any]]:
    """
    Finds and characterizes candidate sources in a 3D SNR cube using DBSCAN.
    This method can distinguish multiple sources at the same sky position but
    at different frequencies (redshifts).

    Args:
        snr_cube: The 3D numpy array of SNR values (y, x, freq).
        snr_threshold: The minimum SNR for a voxel to be considered for clustering.
        eps: DBSCAN `eps` parameter (maximum distance between samples).
        min_samples: DBSCAN `min_samples` parameter (core point neighborhood size).
        freqs_mhz: The array of channel center frequencies in MHz.

    Returns:
        A list of candidate cluster dictionaries, sorted by peak SNR.
    """
    if snr_cube.ndim != 3:
        raise ValueError("snr_cube must be a 3D array.")

    # 1. Threshold the SNR cube to get a list of "hit" voxel coordinates
    hit_coords = np.argwhere(snr_cube > snr_threshold)
    if len(hit_coords) == 0:
        return []

    # 2. Run DBSCAN on the (y, x, z) coordinates of the hits
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(hit_coords)
    labels = db.labels_
    
    unique_labels = set(labels)
    if -1 in unique_labels: unique_labels.remove(-1) # Ignore noise points

    # 3. Process each found cluster to characterize it
    candidates = []
    for label in unique_labels:
        cluster_mask = (labels == label)
        cluster_coords = hit_coords[cluster_mask]
        cluster_snrs = snr_cube[cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2]]

        # 4. Calculate useful properties for this candidate
        peak_snr = np.max(cluster_snrs)
        summed_snr = np.sum(cluster_snrs)
        
        # Calculate the SNR-weighted centroid of the cluster IN INDEX UNITS
        centroid_y = np.sum(cluster_coords[:, 0] * cluster_snrs) / summed_snr
        centroid_x = np.sum(cluster_coords[:, 1] * cluster_snrs) / summed_snr
        centroid_z_idx = np.sum(cluster_coords[:, 2] * cluster_snrs) / summed_snr

        # If freqs_mhz is provided, convert the z-centroid from index to MHz.
        # This uses linear interpolation for sub-channel precision.
        if freqs_mhz is not None:
            if centroid_z_idx < 0 or centroid_z_idx > len(freqs_mhz) - 1:
                # Handle edge cases where centroid is outside the array
                centroid_z_freq = np.interp(centroid_z_idx, np.arange(len(freqs_mhz)), freqs_mhz, left=freqs_mhz[0], right=freqs_mhz[-1])
            else:
                centroid_z_freq = np.interp(centroid_z_idx, np.arange(len(freqs_mhz)), freqs_mhz)
        else:
            # If no frequency axis is given, default to the index (original buggy behavior)
            centroid_z_freq = centroid_z_idx
        # ----------------------

        candidates.append({
            'label': label,
            'peak_snr': peak_snr,
            'integrated_snr': np.sqrt(np.sum(cluster_snrs**2)),
            'size_voxels': len(cluster_coords),
            'centroid_y': centroid_y,
            'centroid_x': centroid_x,
            'centroid_z_freq': centroid_z_freq, # This now correctly holds the frequency in MHz
            'coords': cluster_coords,
        })

    return sorted(candidates, key=lambda c: c['peak_snr'], reverse=True)


def match_candidates_to_truth_3d(
    candidates: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    grid_shape: Tuple[int, int],
    freqs_mhz: np.ndarray,
    spatial_radius_pix: float = 2.0,
    freq_radius_mhz: float = 5.0
) -> Dict[str, List]:
    """
    Matches found candidates to a ground truth list based on their proximity
    in all three dimensions (y, x, frequency).

    A candidate is matched to a true injection if its centroid is within a given
    spatial radius (in pixels) and frequency radius (in MHz) of the true signal.
    Each injection can only be matched to one candidate, and vice-versa.

    Args:
        candidates: The list of candidate dictionaries found by DBSCAN.
        ground_truth: The dictionary containing the list of true injected signals.
        grid_shape: The (ny, nx) shape of the sky grid.
        freqs_mhz: The array of channel center frequencies in MHz.
        spatial_radius_pix: The maximum 2D distance on the sky for a match.
        freq_radius_mhz: The maximum frequency distance for a match.

    Returns:
        A dictionary containing three lists: 'true_positives', 'false_positives',
        and 'false_negatives'.
    """
    injections = ground_truth.get('injections', [])
    if not candidates:
        return {'true_positives': [], 'false_positives': [], 'false_negatives': injections}
    if not injections:
        return {'true_positives': [], 'false_positives': candidates, 'false_negatives': []}

    # Use sets for efficient tracking of matched indices
    matched_cand_indices = set()
    matched_inj_indices = set()
    true_positives = []

    # Pre-calculate coordinates for all true injections
    injections_with_coords = []
    for inj in injections:
        inj_y, inj_x = np.unravel_index(inj['pixel_index'], grid_shape)
        inj_freq = oss.z_to_freq(inj['z'])
        injections_with_coords.append({'inj': inj, 'y': inj_y, 'x': inj_x, 'freq': inj_freq})

    # --- Robust Matching Loop ---
    for i, cand in enumerate(candidates):
        # This is the corrected logic: use the frequency directly from the candidate
        cand_freq = cand['centroid_z_freq']
        cand_y, cand_x = cand['centroid_y'], cand['centroid_x']
        
        # Find the closest unmatched injection to this candidate
        for j, true_inj_obj in enumerate(injections_with_coords):
            # Skip injections that have already been matched to another candidate
            if j in matched_inj_indices:
                continue

            # Check spatial and frequency distance
            dist_2d = np.sqrt((cand_y - true_inj_obj['y'])**2 + (cand_x - true_inj_obj['x'])**2)
            dist_freq = np.abs(cand_freq - true_inj_obj['freq'])
            
            if dist_2d <= spatial_radius_pix and dist_freq <= freq_radius_mhz:
                # We have a match!
                true_positives.append({'cand': cand, 'inj': true_inj_obj['inj']})
                matched_cand_indices.add(i)
                matched_inj_indices.add(j)
                # Break the inner loop since this candidate is now matched
                break 

    # --- Compile final lists based on matched indices ---
    false_positives = [cand for i, cand in enumerate(candidates) if i not in matched_cand_indices]
    false_negatives = [inj for j, inj in enumerate(injections) if j not in matched_inj_indices]

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }

    
# =============================================================================
# --- Candidate Fitting ---
# =============================================================================

def freq_to_idx(freq_axis_mhz: np.ndarray, target_freq_mhz: float) -> int:
    """
    Converts a frequency in MHz to the closest integer channel index.

    Args:
        freq_axis_mhz: The array of frequency channels in MHz.
        target_freq_mhz: The target frequency in MHz to find.

    Returns:
        The integer index of the channel closest to the target frequency.
    """
    # Find the index of the minimum absolute difference
    return np.argmin(np.abs(freq_axis_mhz - target_freq_mhz))


def gaussian_model(x: np.ndarray, amp: float, mean: float, stddev: float) -> np.ndarray:
    """A simple 1D Gaussian model."""
    return amp * np.exp(-0.5 * ((x - mean) / stddev)**2)


def fit_candidate_gaussian(
    candidate: Dict[str, Any],
    data_cube: np.ndarray,
    freqs_mhz: np.ndarray,
    noise_spectrum: np.ndarray,
    fit_padding_channels: int = 20,
    plot_diagnostics: bool = False
) -> Tuple[Dict, Dict]:
    """
    Fits a simple Gaussian profile to a candidate's spectrum.

    This is a simplified diagnostic fit to recover basic parameters like
    amplitude, center frequency, and width (stddev).

    Parameters
    ----------
    candidate : Dict
        The candidate dictionary, containing location info.
    data_cube : np.ndarray
        The 2D data cube (filtered) to extract the spectrum from.
    freqs_mhz : np.ndarray
        The frequency axis in MHz.
    noise_spectrum : np.ndarray
        The 1D per-channel noise standard deviation.
    fit_padding_channels : int, optional
        The number of channels on each side of the candidate's peak to use
        for the fit.

    Returns
    -------
    Tuple[Dict, Dict]
        - A dictionary of the best-fit parameters ('amp', 'mean', 'stddev').
        - A dictionary of the 1-sigma errors on those parameters.
        Returns two empty dictionaries if the fit fails.
    """
    try:
        # --- 1. Extract Data for Fitting ---
        # Get pixel index (e.g., from 'centroid_y')
        pixel_idx = int(candidate['centroid_y'])

        # Get the frequency value from the candidate and convert it to the
        # closest integer channel index using our helper function.
        centroid_freq = candidate['centroid_z_freq']
        freq_idx = freq_to_idx(freqs_mhz, centroid_freq)

        spectrum = data_cube[pixel_idx, :]
        start = max(0, freq_idx - fit_padding_channels)
        end = min(len(freqs_mhz), freq_idx + fit_padding_channels)
        
        freqs_fit = freqs_mhz[start:end]
        spectrum_fit = spectrum[start:end]
        noise_fit = noise_spectrum[start:end]
        
        # (The rest of the function is the same...)
        p0 = [spectrum_fit.max(), freqs_mhz[freq_idx], 0.2]
        lower_bounds = [0, -np.inf, 1e-3]; upper_bounds = [np.inf, np.inf, np.inf]

        if plot_diagnostics:
            plt.figure(figsize=(8, 5))
            plt.errorbar(freqs_fit, spectrum_fit, yerr=noise_fit, fmt='o', label='Data to Fit', capsize=3)
            guess_curve = gaussian_model(freqs_fit, *p0)
            plt.plot(freqs_fit, guess_curve, 'r--', label='Initial Guess')
            plt.title(f"Diagnostic for Candidate at Pixel {pixel_idx}, Freq ~{p0[1]:.2f} MHz")
            plt.xlabel("Frequency (MHz)"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True)
            plt.show()

        popt, pcov = curve_fit(
            gaussian_model, xdata=freqs_fit, ydata=spectrum_fit, p0=p0,
            sigma=noise_fit, absolute_sigma=True, maxfev=5000,
            bounds=(lower_bounds, upper_bounds)
        )
        
        chi2 = np.sum(((spectrum_fit - gaussian_model(freqs_fit, *popt)) / noise_fit)**2)
        dof = len(spectrum_fit) - len(popt)
        
        fit_params = {'amp': popt[0], 'mean': popt[1], 'stddev': abs(popt[2]), 'chi2_red': chi2 / dof if dof > 0 else np.inf}
        fit_errs = {'amp_err': np.sqrt(pcov[0,0]), 'mean_err': np.sqrt(pcov[1,1]), 'stddev_err': np.sqrt(pcov[2,2])}
        
        return fit_params, fit_errs

    except (RuntimeError, KeyError, ValueError) as e:
        if plot_diagnostics:
            print(f"--- Fit FAILED. Error: {e} ---")
        return {}, {}


def fit_candidate_forward_model(
    candidate: Dict[str, Any],
    data_cube: np.ndarray,
    freqs_mhz: np.ndarray,
    noise_spectrum: np.ndarray,
    delay_filter_func: callable, # Pass the filter function itself
    delay_cut_ns: float,
    fit_padding_channels: int = 20,
    plot_diagnostics: bool = False
) -> Tuple[Dict, Dict]:
    """
    Fits a candidate using a forward model of a Gaussian convolved with the
    delay filter. This is the most accurate fitting method.
    """
    try:
        # --- 1. Extract Data for Fitting (same as before) ---
        pixel_idx = int(candidate['centroid_y'])
        freq_idx = freq_to_idx(freqs_mhz, candidate['centroid_z_freq'])
        spectrum = data_cube[pixel_idx, :]
        start = max(0, freq_idx - fit_padding_channels)
        end = min(len(freqs_mhz), freq_idx + fit_padding_channels)
        freqs_fit = freqs_mhz[start:end]
        spectrum_fit = spectrum[start:end]
        noise_fit = noise_spectrum[start:end]
        if spectrum_fit.size < 3: return {}, {}

        # --- 2. Define the Forward Model for the Fitter ---
        # This nested function generates a clean Gaussian, filters it,
        # and returns the appropriate slice for comparison.
        def filtered_gaussian_model(x_slice, amp, mean, stddev):
            # a. Generate the clean, intrinsic Gaussian on the FULL frequency axis
            clean_model_full = gaussian_model(freqs_mhz, amp, mean, stddev)
            
            # b. Filter this ideal model with the SAME pipeline filter
            filtered_model_full = delay_filter_func(
                spectrum=clean_model_full,
                weights=np.ones_like(freqs_mhz),
                freqs_mhz=freqs_mhz,
                delay_cut_ns=delay_cut_ns
            )
            
            # c. Return the slice corresponding to the data being fit
            return filtered_model_full[start:end]

        # --- 3. Perform the Fit ---
        p0 = [spectrum_fit.max(), freqs_mhz[freq_idx], 0.2]
        lower_bounds = [0, -np.inf, 1e-3]; upper_bounds = [np.inf, np.inf, np.inf]
        
        popt, pcov = curve_fit(
            filtered_gaussian_model, # Use our new forward model
            xdata=freqs_fit, ydata=spectrum_fit, p0=p0,
            sigma=noise_fit, absolute_sigma=True, maxfev=5000,
            bounds=(lower_bounds, upper_bounds)
        )
        
        chi2 = np.sum(((spectrum_fit - gaussian_model(freqs_fit, *popt)) / noise_fit)**2)
        dof = len(spectrum_fit) - len(popt)
        
        fit_params = {'amp': popt[0], 'mean': popt[1], 'stddev': abs(popt[2]), 'chi2_red': chi2 / dof if dof > 0 else np.inf}
        fit_errs = {'amp_err': np.sqrt(pcov[0,0]), 'mean_err': np.sqrt(pcov[1,1]), 'stddev_err': np.sqrt(pcov[2,2])}
        
        return fit_params, fit_errs

    except (RuntimeError, KeyError, ValueError) as e:
        if plot_diagnostics:
            print(f"--- Fit FAILED. Error: {e} ---")
        return {}, {}


# =============================================================================
# --- Performance Assessment ---
# =============================================================================

def calculate_performance_stats(matched_results: Dict) -> Dict:
    """Calculates summary statistics from matched results."""
    n_tp = len(matched_results['true_positives'])
    n_fp = len(matched_results['false_positives'])
    n_fn = len(matched_results['false_negatives'])
    n_total_injections = n_tp + n_fn
    n_total_candidates = n_tp + n_fp
    
    completeness = n_tp / n_total_injections if n_total_injections > 0 else 0
    purity = n_tp / n_total_candidates if n_total_candidates > 0 else 0
    
    stats = {
        'num_true_positives': n_tp,
        'num_false_positives': n_fp,
        'num_false_negatives': n_fn,
        'completeness': completeness,
        'purity': purity,
    }
    print("\n--- Search Performance Report ---")
    print(f"Completeness (found / total injected): {completeness:.2%} ({n_tp}/{n_total_injections})")
    print(f"Purity (real / total found):         {purity:.2%} ({n_tp}/{n_total_candidates})")
    print("-" * 33)
    return stats

    
# =============================================================================
# --- Visualization Suite ---
# =============================================================================


def plot_snr_map(
    snr_map: np.ndarray,
    title: str = 'SNR Map',
    vmax_percentile: float = 99.9,
    cmap: Any = cmocean.cm.thermal
) -> None:
    """Plots a 2D SNR map with a robust color scale."""
    plt.figure(figsize=(10, 8))
    vmax = np.percentile(snr_map[np.isfinite(snr_map)], vmax_percentile)
    plt.imshow(snr_map, cmap=cmap, origin='lower', aspect='auto', vmax=vmax)
    plt.colorbar(label='Signal-to-Noise Ratio (SNR)')
    plt.title(title)
    plt.xlabel('Pixel X-coordinate')
    plt.ylabel('Pixel Y-coordinate')
    plt.show()

def plot_spectrum(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    title: str = 'Spectrum',
    original_spectrum: Optional[np.ndarray] = None
) -> None:
    """Plots a spectrum, with an optional unfiltered original for comparison."""
    plt.figure(figsize=(12, 6))
    if original_spectrum is not None:
        plt.plot(freqs, original_spectrum, color='gray', alpha=0.5, label='Original Data')
    plt.plot(freqs, spectrum, color='C0', label='Filtered Data')
    plt.title(title)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Flux (arbitrary units)')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.show()

    
def plot_performance_overview(
    snr_map: np.ndarray,
    grid_shape: Tuple[int, int],
    matched_results: Dict,
    title: str = 'Search Performance Overview'
) -> None:
    """
    Creates a summary plot showing locations of true/false positives/negatives.
    """
    ny, nx = grid_shape
    
    # Get coordinates for each category
    tp_inj_coords = np.array([np.unravel_index(m['inj']['pixel_index'], (ny, nx)) for m in matched_results['true_positives']])
    fp_coords = np.array([[m['centroid_y'], m['centroid_x']] for m in matched_results['false_positives']])
    fn_coords = np.array([np.unravel_index(m['pixel_index'], (ny, nx)) for m in matched_results['false_negatives']])
    
    plt.figure(figsize=(12, 10))
    # vmax = np.percentile(snr_map, 99.9)
    # vmax = 15.0
    plt.imshow(snr_map, cmap=cmocean.cm.thermal, origin='lower', aspect='auto')
    plt.colorbar(label='Max SNR')
    
    # Plot markers for each category
    if len(fn_coords) > 0:
        plt.scatter(fn_coords[:, 1], fn_coords[:, 0], s=150, facecolors='none', edgecolors='red', lw=2, label=f'False Negatives ({len(fn_coords)})')
    if len(tp_inj_coords) > 0:
        plt.scatter(tp_inj_coords[:, 1], tp_inj_coords[:, 0], s=120, marker='o', edgecolors='cyan', facecolor='none', label=f'True Positives ({len(tp_inj_coords)})')
    if len(fp_coords) > 0:
        plt.scatter(fp_coords[:, 1], fp_coords[:, 0], s=120, marker='x', c='lime', label=f'False Positives ({len(fp_coords)})')

    plt.legend(loc='upper right', facecolor='white', framealpha=0.8)
    plt.title(title)
    plt.xlabel('Pixel X-coordinate')
    plt.ylabel('Pixel Y-coordinate')
    plt.show()
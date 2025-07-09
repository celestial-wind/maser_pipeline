# ohm_candidate_finder.py (Regenerated Version)

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
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import cmocean

import ohm_search_simulator as oss # <--- ADD THIS LINE

# =============================================================================
# --- Candidate Finding Algorithms ---
# =============================================================================

# def find_candidates_2d(snr_map: np.ndarray, snr_threshold: float) -> List[Dict[str, Any]]:
#     """
#     Finds candidate sources by applying a simple threshold to a 2D SNR map.

#     Args:
#         snr_map: A 2D numpy array representing the maximum SNR in each pixel.
#         snr_threshold: The minimum SNR to be considered a candidate.

#     Returns:
#         A list of candidate dictionaries, each with pixel coordinates and SNR.
#     """
#     if snr_map.ndim != 2:
#         raise ValueError("snr_map must be a 2D array.")
        
#     hit_coords = np.argwhere(snr_map > snr_threshold)
#     if len(hit_coords) == 0:
#         return []

#     candidates = []
#     for y, x in hit_coords:
#         candidates.append({
#             'peak_snr': snr_map[y, x],
#             'centroid_y': y,
#             'centroid_x': x,
#             'coords': np.array([[y, x]]) # Keep format consistent with 3D finder
#         })
        
#     return sorted(candidates, key=lambda c: c['peak_snr'], reverse=True)

# # In ohm_candidate_finder.py, replace find_candidates_2d

def find_candidates_2d(
    snr_map: np.ndarray,
    snr_threshold: float,
    freq_idx_map: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Finds candidate sources by thresholding a 2D SNR map.
    MODIFIED: Now optionally accepts a map of the frequency index of the peak SNR.
    """
    if snr_map.ndim != 2:
        raise ValueError("snr_map must be a 2D array.")
        
    hit_coords = np.argwhere(snr_map > snr_threshold)
    if len(hit_coords) == 0:
        return []

    candidates = []
    for y, x in hit_coords:
        cand = {
            'peak_snr': snr_map[y, x],
            'centroid_y': y,
            'centroid_x': x,
            'coords': np.array([[y, x]])
        }
        # If the frequency map is provided, add the peak frequency index
        if freq_idx_map is not None:
            cand['peak_freq_idx'] = freq_idx_map[y, x]
        candidates.append(cand)
        
    return sorted(candidates, key=lambda c: c['peak_snr'], reverse=True)

# def find_candidates_3d_dbscan(
#     snr_cube: np.ndarray,
#     snr_threshold: float,
#     eps: int = 3,
#     min_samples: int = 5
# ) -> List[Dict[str, Any]]:
#     """
#     Finds and characterizes candidate sources in a 3D SNR cube using DBSCAN.
#     This method can distinguish multiple sources at the same sky position but
#     at different frequencies (redshifts).

#     Args:
#         snr_cube: The 3D numpy array of SNR values (y, x, freq).
#         snr_threshold: The minimum SNR for a voxel to be considered for clustering.
#         eps: DBSCAN `eps` parameter (maximum distance between samples).
#         min_samples: DBSCAN `min_samples` parameter (core point neighborhood size).

#     Returns:
#         A list of candidate cluster dictionaries, sorted by peak SNR.
#     """
#     if snr_cube.ndim != 3:
#         raise ValueError("snr_cube must be a 3D array.")

#     # 1. Threshold the SNR cube to get a list of "hit" voxel coordinates
#     hit_coords = np.argwhere(snr_cube > snr_threshold)
#     if len(hit_coords) == 0:
#         return []

#     # 2. Run DBSCAN on the (y, x, z) coordinates of the hits
#     db = DBSCAN(eps=eps, min_samples=min_samples).fit(hit_coords)
#     labels = db.labels_
    
#     unique_labels = set(labels)
#     if -1 in unique_labels: unique_labels.remove(-1) # Ignore noise points

#     # 3. Process each found cluster to characterize it
#     candidates = []
#     for label in unique_labels:
#         cluster_mask = (labels == label)
#         cluster_coords = hit_coords[cluster_mask] # List of (y,x,z) for this cluster
#         cluster_snrs = snr_cube[cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2]]

#         # 4. Calculate useful properties for this candidate
#         peak_snr = np.max(cluster_snrs)
#         summed_snr = np.sum(cluster_snrs)
        
#         # Calculate the SNR-weighted centroid of the cluster
#         centroid_y = np.sum(cluster_coords[:, 0] * cluster_snrs) / summed_snr
#         centroid_x = np.sum(cluster_coords[:, 1] * cluster_snrs) / summed_snr
#         centroid_z = np.sum(cluster_coords[:, 2] * cluster_snrs) / summed_snr

#         candidates.append({
#             'label': label,
#             'peak_snr': peak_snr,
#             'integrated_snr': np.sqrt(np.sum(cluster_snrs**2)),
#             'size_voxels': len(cluster_coords),
#             'centroid_y': centroid_y,
#             'centroid_x': centroid_x,
#             'centroid_z_freq': centroid_z, # Centroid in frequency/redshift axis
#             'coords': cluster_coords,
#         })

#     return sorted(candidates, key=lambda c: c['peak_snr'], reverse=True)

#
# CORRECTED find_candidates_3d_dbscan function
#
def find_candidates_3d_dbscan(
    snr_cube: np.ndarray,
    snr_threshold: float,
    eps: int = 3,
    min_samples: int = 5,
    freqs_mhz: np.ndarray = None  # Add freqs_mhz as an optional argument
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

        # --- THIS IS THE FIX ---
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


# def match_candidates_to_truth_3d(
#     candidates_3d: List[Dict],
#     ground_truth: Dict,
#     grid_shape: Tuple[int, int],
#     freqs_mhz: np.ndarray,
#     spatial_radius_pix: int = 5,
#     freq_radius_mhz: float = 10.0
# ) -> Dict:
#     """
#     Matches 3D DBSCAN candidates to a ground truth list in all three dimensions.
#     """
#     true_injections = ground_truth.get('injections', [])
#     if not true_injections:
#         return {'true_positives': [], 'false_positives': candidates_3d, 'false_negatives': []}

#     true_positives = []
#     unmatched_candidates = list(candidates_3d)
    
#     # Create a list of true injections with their coordinates calculated
#     injections_with_coords = []
#     for inj in true_injections:
#         inj_y, inj_x = np.unravel_index(inj['pixel_index'], grid_shape)
#         inj_freq = oss.z_to_freq(inj['z'])
#         injections_with_coords.append({'inj': inj, 'y': inj_y, 'x': inj_x, 'freq': inj_freq})

#     # For each found candidate, see if it matches an injection
#     for cand in candidates_3d:
#         cand_y, cand_x = cand['centroid_y'], cand['centroid_x']
#         cand_freq_idx = cand['centroid_z_freq']
#         cand_freq = freqs_mhz[int(cand_freq_idx)]
        
#         is_matched = False
#         for true_inj_obj in injections_with_coords:
#             # Check spatial distance
#             dist_2d = np.sqrt((cand_y - true_inj_obj['y'])**2 + (cand_x - true_inj_obj['x'])**2)
#             # Check frequency distance
#             dist_freq = np.abs(cand_freq - true_inj_obj['freq'])
            
#             if dist_2d <= spatial_radius_pix and dist_freq <= freq_radius_mhz:
#                 true_positives.append({'cand': cand, 'inj': true_inj_obj['inj']})
#                 injections_with_coords.remove(true_inj_obj) # Matched, so remove from list
#                 unmatched_candidates.remove(cand)
#                 is_matched = True
#                 break
    
#     # False negatives are the true injections that were never matched
#     false_negatives = [item['inj'] for item in injections_with_coords]
    
#     return {
#         'true_positives': true_positives,
#         'false_positives': unmatched_candidates,
#         'false_negatives': false_negatives,
#     }

#
# CORRECTED match_candidates_to_truth_3d function
#
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
# --- Performance Assessment ---
# =============================================================================

def match_candidates_to_truth(
    candidates: List[Dict],
    ground_truth: Dict,
    grid_shape: Tuple[int, int],
    fov_radius_pix: int = 5
) -> Dict:
    """
    Matches found candidates to a ground truth list of injected signals.

    Args:
        candidates: List of candidate dictionaries from a `find_candidates_*` function.
        ground_truth: The ground truth dictionary from the simulator.
        grid_shape: The (y, x) shape of the sky map.
        fov_radius_pix: The search radius in pixels to consider a match.

    Returns:
        A dictionary containing lists of true positives, false positives,
        and false negatives (missed injections).
    """
    true_injections = ground_truth.get('injections', [])
    if not true_injections:
        return {'true_positives': [], 'false_positives': candidates, 'false_negatives': []}

    ny, nx = grid_shape
    true_positives = []
    false_positives = []
    
    # Keep track of which injections and candidates have been matched
    unmatched_injections = list(true_injections)
    unmatched_candidates = list(candidates)

    for cand in candidates:
        cand_y, cand_x = cand['centroid_y'], cand['centroid_x']
        is_matched = False
        
        # Check against all remaining true injections
        for inj in unmatched_injections:
            inj_y, inj_x = np.unravel_index(inj['pixel_index'], (ny, nx))
            dist = np.sqrt((cand_y - inj_y)**2 + (cand_x - inj_x)**2)
            
            if dist <= fov_radius_pix:
                # We have a match!
                true_positives.append({'cand': cand, 'inj': inj})
                unmatched_injections.remove(inj) # Remove from consideration
                unmatched_candidates.remove(cand)
                is_matched = True
                break # Move to the next candidate
        
        if not is_matched:
            # This candidate did not match any remaining injections
            false_positives.append(cand)
            
    # Any injections left in the unmatched list are false negatives
    false_negatives = unmatched_injections
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }

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
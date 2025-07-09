# ohm_fitter.py

"""
A module for characterizing OHM candidates by fitting a physical, 
pipeline-aware model to the data.

This module takes candidates found by a detection algorithm (e.g., the Boxcar
search) and performs a more detailed analysis to extract physical parameters
like redshift (z) and amplitude (amp).

The primary workflow is:
1.  Receive a spectrum containing a candidate signal.
2.  Perform a fine-grained grid search around the candidate's approximate
    redshift using the 'pipeline-aware' matched filter.
3.  Calculate the best-fit amplitude based on the best-fit template.
4.  Provide functions to compare these fitted parameters to ground truth values
    and to visualize the fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# Import the existing pipeline modules
import ohm_template_generator as otg
import ohm_search_simulator as oss

def get_best_fit_amplitude(spectrum: np.ndarray, template: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculates the best-fit amplitude of a template to a spectrum, constrained
    to be non-negative.

    The best-fit amplitude 'A' that minimizes the squared difference between
    the data (d) and the model (A*t) is given by the formula:
    A = (d . t) / (t . t), where the dot product is weighted.

    Args:
        spectrum: The 1D data spectrum slice.
        template: The corresponding 1D model template.
        weights: The weights array for the dot product.

    Returns:
        The best-fit amplitude, constrained to be >= 0.
    """
    template_norm = template - np.mean(template)
    
    numerator = np.sum(spectrum * template_norm * weights)
    denominator = np.sum(template_norm**2 * weights)
    
    if denominator == 0:
        return 0.0
        
    # --- FIX: Enforce non-negativity for emission signals ---
    amplitude = numerator / denominator
    return max(0.0, amplitude)

# def fit_and_compare_parameters(
#     spectrum_slice: np.ndarray,
#     freqs_slice: np.ndarray,
#     full_freqs_hz: np.ndarray,
#     rfi_weights_slice: np.ndarray,
#     noise_per_channel_slice: np.ndarray,
#     intrinsic_template_v: np.ndarray,
#     vel_axis_kms: np.ndarray,
#     ground_truth_injection: Dict[str, Any]
# ) -> Dict[str, Any]:
#     """
#     Fits for z and amplitude and compares them to the ground truth.

#     Args:
#         spectrum_slice: The slice of the data spectrum containing the signal.
#         freqs_slice: The frequency axis corresponding to the slice.
#         full_freqs_hz: The full instrument frequency axis in Hz.
#         rfi_weights_slice: The RFI weights for the spectrum slice.
#         noise_per_channel_slice: The noise sigma for each channel in the slice.
#         intrinsic_template_v: The master high-resolution velocity template.
#         vel_axis_kms: The velocity axis for the master template.
#         ground_truth_injection: The ground truth dictionary for the injected signal.

#     Returns:
#         A dictionary containing the true and fitted parameters.
#     """
#     print("--- Fitting Candidate Parameters ---")
    
#     # 1. Define a fine-grained search grid for redshift around the true value
#     z_true = ground_truth_injection['z']
#     z_search_grid_fine = np.linspace(z_true - 0.05, z_true + 0.05, 100)
    
#     # 2. Perform a fine-grained matched-filter search to find the best-fit redshift
#     #    This is equivalent to a grid search maximizing SNR (minimizing chi-squared).
#     #    We need to pass dummy arrays for the parts of the spectrum we aren't using.
#     dummy_full_spectrum = np.zeros_like(full_freqs_hz)
#     dummy_full_spectrum[:len(spectrum_slice)] = spectrum_slice # This is an approximation
#     dummy_full_rfi = np.ones_like(full_freqs_hz)
#     dummy_full_rfi[:len(rfi_weights_slice)] = rfi_weights_slice
#     dummy_full_noise = np.ones_like(full_freqs_hz) * np.mean(noise_per_channel_slice)
#     dummy_full_noise[:len(noise_per_channel_slice)] = noise_per_channel_slice
    
    
#     best_snr, z_fit, best_template = oss.run_pipeline_aware_matched_filter(
#         dummy_full_spectrum, 
#         full_freqs_hz / 1e6, # Convert back to MHz for function
#         intrinsic_template_v, 
#         vel_axis_kms,
#         dummy_full_rfi,
#         dummy_full_noise,
#         z_search_grid_fine
#     )
    
#     # 3. With the best-fit template, calculate the best-fit amplitude
#     amp_fit = get_best_fit_amplitude(spectrum_slice, best_template, rfi_weights_slice)
    
#     # 4. Compare with ground truth
#     amp_true = ground_truth_injection['amp']
    
#     print("Parameter       |   True   |   Fitted ")
#     print("----------------|----------|----------")
#     print(f"Redshift (z)    | {z_true:<8.4f} | {z_fit:<8.4f}")
#     print(f"Amplitude (amp) | {amp_true:<8.2f} | {amp_fit:<8.2f}")
    
#     results = {
#         'z_true': z_true, 'z_fit': z_fit,
#         'amp_true': amp_true, 'amp_fit': amp_fit,
#         'best_fit_model': best_template * amp_fit
#     }
#     return results

# In ohm_fitter.py, replace the fit_and_compare_parameters function

# In ohm_fitter.py, replace the fit_and_compare_parameters function

# In ohm_fitter.py, replace the fit_and_compare_parameters function

def fit_and_compare_parameters(
    spectrum_slice: np.ndarray,
    full_freqs_hz: np.ndarray,
    rfi_weights_full: np.ndarray,
    noise_per_channel_full: np.ndarray,
    intrinsic_template_v: np.ndarray,
    vel_axis_kms: np.ndarray,
    ground_truth_injection: Dict[str, Any],
    true_start_idx: int, # New argument
    true_end_idx: int    # New argument
) -> Dict[str, Any]:
    """
    Fits for z and amplitude and compares them to the ground truth.
    MODIFIED: Now calculates the expected amplitude after filter suppression.
    """
    print("--- Fitting Candidate Parameters ---")
    # ... (dummy_full_spectrum and z_search_grid_fine are the same) ...
    dummy_full_spectrum = np.zeros_like(full_freqs_hz)
    dummy_full_spectrum[true_start_idx:true_end_idx] = spectrum_slice
    z_true = ground_truth_injection['z']
    z_search_grid_fine = np.linspace(z_true - 0.05, z_true + 0.05, 100)

    best_snr, z_fit, best_template, start_fit, end_fit = oss.run_fine_fit_search(
        dummy_full_spectrum, full_freqs_hz / 1e6, intrinsic_template_v,
        vel_axis_kms, rfi_weights_full, noise_per_channel_full, z_search_grid_fine
    )
    
    if best_template is None:
        print("Fit failed to produce a valid template.")
        return {}

    # --- NEW: Calculate the filter suppression factor ---
    # 1. Generate the ideal, un-filtered template at the best-fit redshift
    ideal_template, _, _ = otg.process_to_native_resolution(
        intrinsic_template_v, vel_axis_kms, z_fit, (full_freqs_hz / 1e6)
    )
    # 2. The suppression is the ratio of the filtered peak to the ideal peak
    suppression_factor = np.max(best_template) / np.max(ideal_template) if np.max(ideal_template) > 0 else 0
    # --- End of NEW block ---

    fit_spectrum_slice = dummy_full_spectrum[start_fit:end_fit]
    fit_weights_slice = rfi_weights_full[start_fit:end_fit]
    amp_fit = get_best_fit_amplitude(fit_spectrum_slice, best_template, fit_weights_slice)
    
    amp_true = ground_truth_injection['amp']
    expected_amp = amp_true * suppression_factor

    print("Parameter       |   True   | Expected |   Fitted ")
    print("----------------|----------|----------|----------")
    print(f"Redshift (z)    | {z_true:<8.4f} |    --    | {z_fit:<8.4f}")
    print(f"Amplitude (amp) | {amp_true:<8.2f} | {expected_amp:<8.2f} | {amp_fit:<8.2f}")
    
    results = {
        'z_true': z_true, 'z_fit': z_fit,
        'amp_true': amp_true, 'amp_fit': amp_fit, 'amp_expected': expected_amp,
        'best_fit_model': best_template * amp_fit,
        'fit_freqs_slice': (full_freqs_hz / 1e6)[start_fit:end_fit],
        'fit_spectrum_slice': fit_spectrum_slice
    }
    return results
    
def plot_fit_result(
    fit_results: Dict[str, Any],
    title: str = "Maser Parameter Fit"
) -> None:
    """
    Visualizes the data, the best-fit model, and the residuals.
    MODIFIED: Now sources all data directly from the fit_results dictionary
    to ensure self-consistency and prevent shape errors.
    """
    # --- FIX: Unpack all necessary arrays from the results dictionary ---
    if not fit_results:
        print("Fit results are empty, cannot plot.")
        return
        
    best_fit_model = fit_results['best_fit_model']
    spectrum_slice = fit_results['fit_spectrum_slice']
    freqs_slice = fit_results['fit_freqs_slice']
    # --- End of FIX ---

    residuals = spectrum_slice - best_fit_model

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1]})

    # Top panel: Data and Model Fit
    # Keep the 'blocky' step plot as you preferred
    axs[0].step(freqs_slice, spectrum_slice, where='mid', label='Data')
    axs[0].plot(freqs_slice, best_fit_model, 'r-', lw=2, label=f"Best Fit (z={fit_results['z_fit']:.4f})")
    axs[0].legend()
    axs[0].set_ylabel("Flux (arbitrary units)")
    axs[0].set_title(title)
    axs[0].grid(True, alpha=0.5)

    # Bottom panel: Residuals
    axs[1].step(freqs_slice, residuals, where='mid', color='gray') # Changed to step to match
    axs[1].axhline(0, ls='--', color='k')
    axs[1].set_xlabel("Frequency (MHz)")
    axs[1].set_ylabel("Residuals")
    axs[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()
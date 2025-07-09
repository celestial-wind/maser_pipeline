# ohm_fitter.py (Production Version)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import warnings
from typing import Dict, Any

import ohm_search_simulator as oss

C_KMS = 299792.458

def gaussian_with_baseline(x, amplitude, center, sigma, baseline):
    """Gaussian function with baseline offset."""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2) + baseline

def fit_gaussian_peak(x, y, center_guess, window_size=60, smooth_sigma=2.0, plot=True):
    """
    Fits a Gaussian to a peak with a FIXED center and constrained parameters.
    """
    # --- 1. Define the fitting window around the KNOWN center guess ---
    center_idx = np.argmin(np.abs(x - center_guess))
    start_idx = max(0, center_idx - window_size // 2)
    end_idx = min(len(x), center_idx + window_size // 2)
    x_fit, y_fit = x[start_idx:end_idx], y[start_idx:end_idx]
    
    # --- 2. Estimate initial parameters for the free variables ---
    baseline_est = np.median(y_fit)
    amplitude_est = np.max(y_fit) - baseline_est
    sigma_est = 1.0 # Start with a reasonable guess of 1 MHz width

    # --- 3. FIX THE CENTER PARAMETER ---
    # We create a new, simpler model function where 'center' is no longer
    # a free parameter, but is fixed to our 'center_guess'.
    model_to_fit = lambda x_lambda, amplitude, sigma, baseline: \
        gaussian_with_baseline(x_lambda, amplitude, center_guess, sigma, baseline)

    # --- 4. Set up robust parameter bounds ---
    # The bounds now only apply to the 3 free parameters: amp, sigma, baseline
    
    # Convert physical velocity width limit to frequency width in MHz
    # FWHM < 1000 km/s  --> sigma < (1000 / 2.355) km/s
    # sigma_MHz = sigma_kms * center_MHz / c
    sigma_max_kms = 1000.0 / 2.355
    sigma_max_mhz = sigma_max_kms * center_guess / C_KMS
    
    lower_bounds = [0, 0.1, -np.inf] # Amp > 0, Sigma > 0.1 MHz
    upper_bounds = [amplitude_est * 2, sigma_max_mhz, np.inf] # Amp < 2*guess, Sigma < 1000 km/s equiv
    bounds = (lower_bounds, upper_bounds)
    
    initial_guess = [amplitude_est, sigma_est, baseline_est]
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit the new model with only 3 free parameters
            popt, pcov = curve_fit(
                model_to_fit, x_fit, y_fit, p0=initial_guess,
                bounds=bounds, maxfev=5000
            )
        # Re-assemble the full parameter list with the fixed center
        fit_params = [popt[0], center_guess, popt[1], popt[2]] # amp, center, sigma, baseline
        param_errors = np.sqrt(np.diag(pcov))
        
        y_model = gaussian_with_baseline(x_fit, *fit_params)
        residuals = y_fit - y_model
        noise_std = np.std(residuals)
        snr = abs(fit_params[0]) / noise_std if noise_std > 0 else 0
        
        results = {
            'success': True, 'amplitude': fit_params[0], 'center': fit_params[1], 
            'sigma': fit_params[2], 'baseline': fit_params[3],
            'amplitude_err': param_errors[0], 'sigma_err': param_errors[1],
            'baseline_err': param_errors[2],
            'fwhm': fit_params[2] * 2.355, 'snr': snr,
            'x_fit': x_fit, 'y_fit': y_fit, 'y_model': y_model
        }
    except Exception as e:
        print(f"Fitting failed: {e}")
        return {'success': False, 'error': str(e)}
    
    if plot and results['success']:
        # --- 5. Create a ZOOMED-IN diagnostic plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data and the best-fit model
        ax.step(x_fit, y_fit, where='mid', label='Data in Fit Window')
        ax.plot(x_fit, y_model, 'r-', linewidth=2, label='Gaussian Fit')
        ax.axhline(fit_params[3], color='gray', linestyle='--', alpha=0.8, label=f'Baseline')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Intensity')
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(f'Zoomed-in Gaussian Fit (SNR={snr:.1f})')
        plt.show()
        
    return results


def analyze_candidate_fit(
    full_pixel_spectrum: np.ndarray,
    full_freqs_mhz: np.ndarray,
    candidate: Dict[str, Any],
    ground_truth_injection: Dict[str, Any]
) -> None:
    """
    Wrapper function that uses the robust `fit_gaussian_peak` to analyze
    a candidate from our simulation and compares the result to ground truth.
    """
    print("--- Fitting Candidate Parameters (Constrained Routine) ---")

    center_guess_mhz = full_freqs_mhz[candidate['peak_freq_idx']]

    fit_results = fit_gaussian_peak(
        x=full_freqs_mhz,
        y=full_pixel_spectrum,
        center_guess=center_guess_mhz,
        plot=True # Make sure the plot is generated
    )

    if fit_results['success']:
        z_fit = oss.freq_to_z(fit_results['center'])
        fwhm_fit_kms = C_KMS * (fit_results['fwhm'] / fit_results['center'])
        amp_fit = fit_results['amplitude']

        z_true = ground_truth_injection['z']
        amp_true = ground_truth_injection['amp']

        print("\n--- Parameter Comparison ---")
        print("Parameter         |   True   |   Fitted ")
        print("------------------|----------|----------")
        print(f"Redshift (z)      | {z_true:<8.4f} | {z_fit:<8.4f}")
        print(f"Amplitude (amp)   | {amp_true:<8.2f} | {amp_fit:<8.2f}")
        print(f"FWHM (km/s)       |    --    | {fwhm_fit_kms:<8.1f}")
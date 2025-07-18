# ohm_template_generator.py

"""
A comprehensive module for generating synthetic OH Megamaser (OHM) profiles 
and creating matched-filter templates for astronomical searches.

This module provides functions to:
1. Generate high-resolution, physically-motivated OHM profiles in velocity space.
2. Average a large population of these profiles to create a robust "optimal"
   intrinsic template.
3. Process the intrinsic template for a specific redshift and instrumental setup,
   simulating effects like spectral smoothing and rebinning.
4. Generate simple, generic templates (e.g., Boxcar, Gaussian) for baseline
   search performance comparisons.
"""

import numpy as np
from scipy.signal import windows
from tqdm.auto import tqdm
from typing import Tuple, List, Optional

# =============================================================================
# --- Constants ---
# =============================================================================

C_KMS = 299792.458  # Speed of light in km/s
NU_1667_REST = 1667.359  # Rest frequency of 1667 MHz line in MHz
NU_1665_REST = 1665.402  # Rest frequency of 1665 MHz line in MHz

# Example CHIME-like instrumental parameters
CHIME_FREQS = np.linspace(400, 800, 1024)
NATIVE_CHANNEL_WIDTH = np.mean(np.diff(CHIME_FREQS))  # ~0.39 MHz

# =============================================================================
# --- Core Utility Functions ---
# =============================================================================


def fwhm_to_sigma(fwhm: float) -> float:
    """Converts a Full-Width at Half-Maximum (FWHM) to a standard deviation."""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def gaussian(x: np.ndarray, mu: float, sigma: float, amplitude: float = 1.0) -> np.ndarray:
    """Computes a Gaussian (normal) distribution."""
    if sigma <= 0:
        return np.zeros_like(x)
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def apply_pfb_smoothing(spectrum: np.ndarray, kernel_width: int) -> np.ndarray:
    """
    Simulates the local spectral leakage of a Polyphase Filter Bank (PFB)
    by convolving the spectrum with a sinc-Hamming window. [squared for frquency response]

    Args:
        spectrum: The 1D spectrum to be smoothed.
        kernel_width: The width of the smoothing kernel in array elements.

    Returns:
        The smoothed 1D spectrum.
    """
    if kernel_width < 2: return spectrum
    kernel = create_chime_pfb_window_direct(len(spectrum), ntap=4)
    kernel /= np.sum(kernel)
    return np.convolve(spectrum, kernel, mode='same')


def create_chime_pfb_window_direct(N, ntap=4):
    """
    Direct implementation for CHIME; based on known characteristics of the CHIME Polyphase Filter Bank (PFB)
    """
    
    # Channel index relative to center
    k = np.arange(N) - N//2
    
    # Sinc^2 response
    sinc_squared = np.sinc(ntap * k / N) ** 2
    
    # Apply windowing
    hamming_window = windows.hamming(N)
    
    return sinc_squared * hamming_window

    
def rebin_spectrum(spec_hr: np.ndarray, freq_hr: np.ndarray, freqs_native: np.ndarray) -> np.ndarray:
    """
    Bins a high-resolution spectrum down to a native, lower-resolution grid
    using an efficient, vectorized method.

    Args:
        spec_hr: The high-resolution input spectrum values.
        freq_hr: The corresponding high-resolution frequency axis.
        freqs_native: The target native frequency grid.

    Returns:
        The binned spectrum on the native grid.
    """
    if freqs_native.size == 0:
        return np.array([])

    # Define the edges of the native frequency bins
    bin_edges = np.zeros(len(freqs_native) + 1)
    bin_edges[1:-1] = (freqs_native[:-1] + freqs_native[1:]) / 2
    bin_edges[0] = freqs_native[0] - NATIVE_CHANNEL_WIDTH / 2
    bin_edges[-1] = freqs_native[-1] + NATIVE_CHANNEL_WIDTH / 2

    # Digitize the high-res frequencies into the native bins
    digitized = np.digitize(freq_hr, bin_edges)
    
    # Filter out points that fall outside the native bin range
    valid_mask = (digitized > 0) & (digitized <= len(freqs_native))
    
    # Use bincount for highly efficient averaging
    # It sums all values falling into each bin
    min_len = len(freqs_native) + 1 # Ensure bincount output has correct size
    bin_sums = np.bincount(digitized[valid_mask], weights=spec_hr[valid_mask], minlength=min_len)
    bin_counts = np.bincount(digitized[valid_mask], minlength=min_len)

    # Calculate the mean for each bin, avoiding division by zero
    binned_spec = np.zeros_like(freqs_native, dtype=float)
    counts_for_bins = bin_counts[1:] # Index 0 is for points < first edge
    sums_for_bins = bin_sums[1:]
    
    non_empty_bins = counts_for_bins > 0
    binned_spec[non_empty_bins] = sums_for_bins[non_empty_bins] / counts_for_bins[non_empty_bins]

    return binned_spec


# =============================================================================
# --- Intrinsic OHM Profile Generation ---
# =============================================================================


def generate_intrinsic_maser(vel_axis_kms: np.ndarray) -> np.ndarray:
    """
    Generates a single, randomized intrinsic OHM spectrum in velocity space.

    The model is a sum of three Gaussians: two narrow lines (1667 & 1665 MHz)
    and one broader component. Parameters are drawn from log-normal distributions
    to simulate a realistic population of observed masers.

    Args:
        vel_axis_kms: High-resolution velocity axis (km/s) for generation.

    Returns:
        A 1D numpy array of the normalized synthetic OHM profile.
    """
    # Draw physical parameters from log-normal distributions, which are good
    # for modeling quantities that are strictly positive and have a skewed tail.
    fwhm_broad = np.random.lognormal(np.log(250), 0.5) # Broad component FWHM in km/s
    fwhm_1667 = np.random.lognormal(np.log(100), 0.4) # 1667 MHz line FWHM in km/s
    fwhm_1665 = np.random.lognormal(np.log(80), 0.5)  # 1665 MHz line FWHM in km/s
    
    # Ratios of amplitudes
    amp_ratio_1667_over_1665 = np.random.lognormal(np.log(2.0), 0.3)
    amp_ratio_broad_over_1667 = np.random.lognormal(np.log(0.3), 0.5)

    # Convert FWHM to sigma for the Gaussian function
    sigma_broad_kms = fwhm_to_sigma(fwhm_broad)
    sigma_1667_kms = fwhm_to_sigma(fwhm_1667)
    sigma_1665_kms = fwhm_to_sigma(fwhm_1665)
    
    # Velocity separation between the two narrow lines due to their rest frequency difference
    vel_sep_kms = C_KMS * (NU_1667_REST - NU_1665_REST) / NU_1667_REST

    # Set amplitudes based on drawn ratios (1667 line is reference)
    amp_1667 = 1.0
    amp_1665 = amp_1667 / amp_ratio_1667_over_1665
    amp_broad = amp_1667 * amp_ratio_broad_over_1667

    # Generate the three Gaussian components
    spec_1667 = gaussian(vel_axis_kms, 0.0, sigma_1667_kms, amp_1667)
    spec_1665 = gaussian(vel_axis_kms, -vel_sep_kms, sigma_1665_kms, amp_1665)
    spec_broad = gaussian(vel_axis_kms, 0.0, sigma_broad_kms, amp_broad)

    # Combine and normalize the final spectrum
    total_spectrum = spec_1667 + spec_1665 + spec_broad
    if np.max(total_spectrum) > 0:
        total_spectrum /= np.max(total_spectrum)
        
    return total_spectrum

# =============================================================================
# --- Simple Template Generators ---
# =============================================================================


def generate_gaussian_template(width: int, sigma_fraction: float = 0.25) -> np.ndarray:
    """
    Generates a simple Gaussian template centered in an array.

    Args:
        width: The total width of the template array.
        sigma_fraction: The standard deviation as a fraction of the total width.

    Returns:
        A 1D numpy array containing the Gaussian template.
    """
    if width <= 0: return np.array([])
    mu = (width - 1) / 2.0
    sigma = width * sigma_fraction
    x = np.arange(width)
    template = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return template

    
# =============================================================================
# --- Main User-Facing Functions ---
# =============================================================================


def generate_optimal_template(
    N_population: int,
    vel_axis_kms: np.ndarray,
    verbose: bool = False
) -> np.ndarray:
    """
    Creates an optimal average template by simulating and averaging many masers.
    This template is in velocity space and is independent of redshift.

    Args:
        N_population: The number of synthetic masers to average (>5000 recommended).
        vel_axis_kms: The high-resolution velocity axis to generate profiles on.
        verbose: If True, show a tqdm progress bar.

    Returns:
        The high-resolution, intrinsic average template in velocity space.
    """
    if N_population == 0:
        return np.zeros_like(vel_axis_kms)
    
    iterable = range(N_population)
    if verbose:
        iterable = tqdm(iterable, desc="Averaging Maser Population")
        
    # Create a list of many random maser profiles and average them
    population = [generate_intrinsic_maser(vel_axis_kms) for _ in iterable]
    template = np.mean(population, axis=0)
    
    if np.max(template) > 0:
        template /= np.max(template)
        
    return template


def process_to_native_resolution_and_target_z(
    intrinsic_template_v: np.ndarray,
    vel_axis_kms: np.ndarray,
    z: float,
    native_freq_grid: np.ndarray
) -> Tuple[np.ndarray, int, int]:
    """
    Converts a high-res velocity template to a native-resolution frequency template.

    This function performs three key steps:
    1.  Shifts the template to the correct observed frequency for redshift `z`.
    2.  Applies instrumental smoothing to simulate PFB channel leakage.
    3.  Rebins the result down to the telescope's native frequency channels.

    Args:
        intrinsic_template_v: The high-res template in velocity space.
        vel_axis_kms: The corresponding velocity axis for the intrinsic template.
        z: The target redshift for the final template.
        native_freq_grid: The center frequencies of the telescope's native channels.

    Returns:
        A tuple containing:
        - The final template on the native frequency grid.
        - The start index in the full native grid where the template belongs.
        - The end index in the full native grid where the template belongs.
    """
    # 1. Convert the velocity axis to a high-resolution frequency axis at redshift z
    center_freq_obs = NU_1667_REST / (1 + z)
    freq_axis_hr = center_freq_obs * (1 - vel_axis_kms / C_KMS)
    
    # 2. Apply instrumental smoothing (approximates PFB spectral leakage)
    native_channel_width_hz = np.mean(np.diff(native_freq_grid)) * 1e6
    hr_channel_width_hz = np.mean(np.abs(np.diff(freq_axis_hr))) * 1e6
    oversampling_factor = native_channel_width_hz / hr_channel_width_hz if hr_channel_width_hz > 0 else 1
    
    # Determine an appropriate smoothing kernel width based on oversampling
    smoothing_width_hr = int(1.5 * oversampling_factor)
    if smoothing_width_hr % 2 == 0: smoothing_width_hr += 1 # Ensure odd kernel width
    
    smoothed_template = apply_pfb_smoothing(intrinsic_template_v, smoothing_width_hr)

    # 3. Rebin to native frequency resolution
    # Find the slice of the native frequency grid that the signal falls into
    native_indices = np.where(
        (native_freq_grid >= freq_axis_hr.min()) & (native_freq_grid <= freq_axis_hr.max())
    )[0]
    
    if len(native_indices) == 0:
        return np.array([]), -1, -1 # Return empty if signal is out of band

    native_freqs_subset = native_freq_grid[native_indices]
    start_idx, end_idx = native_indices[0], native_indices[-1] + 1
    
    final_template = rebin_spectrum(smoothed_template, freq_axis_hr, native_freqs_subset)
    
    return final_template, start_idx, end_idx

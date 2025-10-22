# ohm_search_simulator.py


"""
The core module for the OHM search simulation pipeline.

This module is responsible for:
1. Simulating a 3D (RA, Dec, Frequency) data cube of the sky.
2. Offering multiple, selectable models for the sky signal (e.g., blank,
   power-law sources, realistic GSM) and instrumental noise (e.g., uniform,
   beam-weighted).
3. Injecting synthetic OHM signals into the data cube.
4. Providing a suite of search algorithms to run on the simulated data,
   ranging from simple thresholding to advanced, "pipeline-aware" matched filters.
"""


# External library dependencies
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Tuple, List, Any, Optional, Union
from scipy.signal import correlate
from scipy.signal.windows import tukey
import healpy as hp
from pygdsm import GlobalSkyModel
from uvtools import dspec


# Local module dependencies
import ohm_template_generator as otg


# =============================================================================
# --- Core Utility Functions ---
# =============================================================================


def z_to_freq(z, rest_freq=1667.359): return rest_freq / (1 + z)
def freq_to_z(freq, rest_freq=1667.359): return (rest_freq / freq) - 1


# =============================================================================
# --- Component Functions: RFI and Foreground Filtering ---
# =============================================================================


def generate_realistic_rfi_mask(freqs: np.ndarray, percentage_random=0.01) -> np.ndarray:
    """
    Generates a stationary RFI mask based on known interfering frequency bands.

    Args:
        freqs: The array of channel center frequencies in MHz.

    Returns:
        A 1D weight array (0 for flagged channels, 1 for clean channels).
    """
    print("  - Generating realistic stationary RFI mask...")
    weights = np.ones_like(freqs)
    
    # Approximate bands for North America. Could be refined.
    rfi_bands_mhz = {
        'LTE_Band_12_17': (698, 716),
        'LTE_Band_13':    (777, 787),
        'Digital_TV_1':   (470, 512),
        'Digital_TV_2':   (524, 608),
    }
    
    for band_name, (start_freq, end_freq) in rfi_bands_mhz.items():
        flag_indices = np.where((freqs >= start_freq) & (freqs <= end_freq))
        if flag_indices[0].size > 0:
            weights[flag_indices] = 0
            
    # Also flag a small fraction of random channels for intermittent RFI
    num_random_flags = int(percentage_random * len(freqs))
    random_indices = np.random.choice(np.where(weights == 1)[0], num_random_flags, replace=False)
    weights[random_indices] = 0
    
    return weights


def infill_rfi(
    spectrum: np.ndarray,
    weights: np.ndarray,
    frequencies: np.ndarray
) -> np.ndarray:
    """
    Infills RFI-flagged regions in a spectrum using a robust power-law fit.

    This function identifies flagged regions (where weight is 0), fits a single
    global power-law model to the valid, positive data points in the spectrum,
    and uses this model to interpolate values for the flagged channels.
    It then adds synthetic noise based on the spectrum's characteristics.

    Parameters
    ----------
    spectrum : np.ndarray
        The 1D array of the spectrum data, potentially containing RFI.
    weights : np.ndarray
        A 1D array of weights. Channels with a weight of 0 are considered
        flagged for RFI and will be infilled.
    frequencies : np.ndarray
        The corresponding frequency axis for the spectrum.

    Returns
    -------
    np.ndarray
        A new spectrum array with the RFI-flagged zones infilled.
    """
    # --- 1. Setup and Input Validation ---
    infilled_spectrum = np.copy(spectrum)
    valid_indices = weights > 0
    flagged_indices = ~valid_indices

    # If there's nothing to infill, return the original spectrum
    if not np.any(flagged_indices):
        return infilled_spectrum

    # --- 2. Robust Power-Law Fit ---
    # To prevent log10 errors, we must fit only to valid AND positive data.
    positive_valid_indices = valid_indices & (spectrum > 0)

    # Check if there are enough positive points for a stable fit
    if np.sum(positive_valid_indices) < 2:
        # Fallback: if no stable fit is possible, just fill with noise
        # calculated from the entire valid spectrum.
        noise_sigma = np.std(spectrum[valid_indices])
        if np.isnan(noise_sigma) or noise_sigma == 0: noise_sigma = 1.0 # Ultimate fallback
        synthetic_noise = np.random.normal(0, noise_sigma, size=np.sum(flagged_indices))
        infilled_spectrum[flagged_indices] = synthetic_noise
        return infilled_spectrum

    # Fit a line (y = k*x + b) in log-log space.
    # y = log10(spectrum), x = log10(frequency)
    log_freq = np.log10(frequencies[positive_valid_indices])
    log_spec = np.log10(spectrum[positive_valid_indices])
    k, log_A = np.polyfit(log_freq, log_spec, 1)

    # Generate the power-law model (A * f^k) for all frequencies
    power_law_model = (10**log_A) * (frequencies**k)

    # --- 3. Infill and Add Noise ---
    # Infill the flagged regions with the smooth power-law model
    infilled_spectrum[flagged_indices] = power_law_model[flagged_indices]

    # Measure the noise from the valid regions by comparing against the model
    residual = spectrum[valid_indices] - power_law_model[valid_indices]
    
    # Calculate noise sigma, ensuring we only use finite numbers
    noise_sigma = np.std(residual[np.isfinite(residual)])

    # Final safety check for the noise value
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        noise_sigma = 1e-6 # Use a tiny floor value if noise is zero or undefined

    # Generate synthetic noise for the flagged channels
    synthetic_noise = np.random.normal(0, noise_sigma, size=np.sum(flagged_indices))

    # Add the synthetic noise to the infilled regions
    infilled_spectrum[flagged_indices] += synthetic_noise

    return infilled_spectrum



def infill_rfi_iterative(
    spectrum: np.ndarray,
    weights: np.ndarray,
    frequencies: np.ndarray,
    n_iter: int = 5,
    sigma_clip: float = 3.0,
    global_noise_sigma: Optional[float] = None
) -> np.ndarray:
    """
    Infills RFI regions using a robust iterative fit and a global noise estimate.

    This version corrects a bug in the iterative fitter and uses an optional
    global noise estimate to prevent bright signals from skewing the
    amplitude of the infilled noise.
    """
    infilled_spectrum = np.copy(spectrum)
    valid_indices = weights > 0
    flagged_indices = ~valid_indices

    if not np.any(flagged_indices):
        return infilled_spectrum

    fit_indices = valid_indices & (spectrum > 0)

    if np.sum(fit_indices) < 2:
        noise_sigma = global_noise_sigma if global_noise_sigma is not None else 1.0
        infilled_spectrum[flagged_indices] = np.random.normal(0, noise_sigma, size=np.sum(flagged_indices))
        return infilled_spectrum

    # Iteratively clip outliers and re-fit the power law
    for i in range(n_iter):
        log_freq = np.log10(frequencies[fit_indices])
        log_spec = np.log10(spectrum[fit_indices])
        
        if len(log_freq) < 2: break
        k, log_A = np.polyfit(log_freq, log_spec, 1)

        current_model_for_clip = (10**log_A) * (frequencies[fit_indices]**k)
        residual = spectrum[fit_indices] - current_model_for_clip
        sigma = np.std(residual)
        if sigma == 0: break

        # This is the corrected logic for updating the fit indices
        is_not_outlier = np.abs(residual) < (sigma_clip * sigma)
        current_fit_global_indices = np.where(fit_indices)[0]
        clean_subset_indices = current_fit_global_indices[is_not_outlier]
        
        new_fit_mask = np.zeros_like(spectrum, dtype=bool)
        new_fit_mask[clean_subset_indices] = True
        fit_indices = new_fit_mask & (spectrum > 0)

    # --- Final Infilling ---
    final_power_law_model = (10**log_A) * (frequencies**k)
    infilled_spectrum[flagged_indices] = final_power_law_model[flagged_indices]

    # Calculate the robust local noise sigma from the clean, clipped data
    final_clean_residual = spectrum[fit_indices] - final_power_law_model[fit_indices]
    local_noise_sigma = np.std(final_clean_residual)

    # --- Use Global Noise Estimate if Local is Anomalous ---
    noise_sigma_to_use = local_noise_sigma
    if global_noise_sigma is not None and local_noise_sigma > (2.0 * global_noise_sigma):
        # If local noise is >2x the global estimate, it's likely skewed.
        # Fall back to the more reliable global value.
        noise_sigma_to_use = global_noise_sigma

    if not np.isfinite(noise_sigma_to_use) or noise_sigma_to_use <= 0:
        noise_sigma_to_use = 1e-6

    synthetic_noise = np.random.normal(0, noise_sigma_to_use, size=np.sum(flagged_indices))
    infilled_spectrum[flagged_indices] += synthetic_noise

    return infilled_spectrum


def find_and_mask_outliers(
    spectrum: np.ndarray,
    sigma_threshold: float = 7.0,
    mask_bandwidth_channels: int = 5
) -> np.ndarray:
    """
    Identifies significant outliers and masks a defined bandwidth around them.

    Args:
        spectrum: The input spectrum.
        sigma_threshold: The N-sigma threshold to identify an outlier peak.
        mask_bandwidth_channels: The number of channels to mask on EACH side of a detected peak.

    Returns:
        A new spectrum with a region around each outlier replaced by the median.
    """
    # Use robust statistics (Median Absolute Deviation) to find outliers
    median_val = np.median(spectrum)
    abs_deviation = np.abs(spectrum - median_val)
    mad = np.median(abs_deviation)
    robust_sigma = mad * 1.4826
    
    # Find the indices of all channels that exceed the outlier threshold
    outlier_indices = np.where(abs_deviation > (sigma_threshold * robust_sigma))[0]
    
    if len(outlier_indices) == 0:
        return spectrum # No outliers found, return the original spectrum

    # Create a boolean mask for the channels to be replaced
    channels_to_mask = np.zeros_like(spectrum, dtype=bool)
    for idx in outlier_indices:
        # Define the window to mask around the outlier
        start = max(0, idx - mask_bandwidth_channels)
        end = min(len(spectrum), idx + mask_bandwidth_channels + 1)
        channels_to_mask[start:end] = True

    # Create a new spectrum and replace the masked regions with the median
    masked_spectrum = np.copy(spectrum)
    masked_spectrum[channels_to_mask] = median_val
    
    return masked_spectrum

    
def apply_dayneu_filter(
    spectrum: np.ndarray,
    frequencies_mhz: np.ndarray,
    delay_cutoff_ns: float,
    weights: np.array =None,
    cache: dict = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies a Dayenu delay filter to a 1D spectrum to remove smooth foregrounds.

    This function serves as a user-friendly wrapper around the `uvtools`
    fourier_filter function, specifically configured for the 'dayenu' mode.
    It filters the data in the delay domain (the Fourier transform of the
    frequency spectrum), removing components within a specified delay range
    centered at zero. This is effective for removing spectrally smooth
* **spectrum** (*np.ndarray*): A 1D numpy array containing the flux or
        amplitude values of the spectrum to be filtered.

* **frequencies_mhz** (*np.ndarray*): A 1D numpy array with the same shape as
        `spectrum`, containing the corresponding frequency for each channel in
        units of **MHz**.

* **delay_cutoff_ns** (*float*): The half-width of the filter in the delay
        domain, specified in **nanoseconds (ns)**. The filter will remove all
        spectral components corresponding to delays between -`delay_cutoff_ns`
        and +`delay_cutoff_ns`. A larger value will remove more aggressive,
        less smooth foreground structures. A good starting point is often
        related to the inverse of your signal's bandwidth.

* **weights** (*np.ndarray, optional*): A 1D numpy array of the same shape as
        `spectrum` that specifies the relative weight of each data point.
        - A weight of **1.0** means the data point is fully trusted.
        - A weight of **0.0** means the data point is flagged (e.g., due to RFI)
          and will be ignored during the fitting process. The filter will
          interpolate the model over these flagged regions.
        - If `None`, the function will assume uniform weights of 1.0 for all
          channels, meaning all data points are trusted equally.

    Returns
    -------
* **filtered_spectrum** (*np.ndarray*): The 1D spectrum after the foreground
        model has been subtracted. This is the "clean" data containing the
        residual signals.

* **foreground_model** (*np.ndarray*): The smooth foreground model that the
        Dayenu filter fitted to the data and subsequently removed.
    """
    # --- 1. Input Validation and Setup ---

    # Ensure inputs are numpy arrays
    spectrum = np.asarray(spectrum)
    frequencies_mhz = np.asarray(frequencies_mhz)

    if spectrum.ndim != 1 or frequencies_mhz.ndim != 1:
        raise ValueError("Input 'spectrum' and 'frequencies_mhz' must be 1D arrays.")

    if spectrum.shape != frequencies_mhz.shape:
        raise ValueError("Input 'spectrum' and 'frequencies_mhz' must have the same shape.")

    # If no weights are provided, assume uniform weights of 1.0 for all channels.
    if weights is None:
        weights = np.ones_like(spectrum)
    else:
        weights = np.asarray(weights)
        if weights.shape != spectrum.shape:
            raise ValueError("Input 'weights' must have the same shape as the spectrum.")

    # The fourier_filter function expects the x-axis (frequency) to be in GHz
    # for the delay units (ns) to be interpreted correctly.
    frequencies_ghz = frequencies_mhz / 1000.0

    # --- 2. Define Filter Parameters ---

    # For a standard delay filter, the region to be filtered is centered at a delay of 0.
    filter_center = [0.]

    # The half-width is specified by the user via the delay_cutoff_ns parameter.
    filter_half_width = [delay_cutoff_ns]

    # --- 3. Apply the Dayenu Filter ---

    # Call the main filtering function from dspec.py.
    # We specify 'dayenu' mode and to filter along the first (and only) dimension.
    foreground_model, filtered_spectrum, _ = dspec.fourier_filter(
        x=frequencies_ghz,
        data=spectrum,
        wgts=weights,
        filter_centers=filter_center,
        filter_half_widths=filter_half_width,
        mode='dayenu',
        filter_dims=1,
        cache=cache
    )

    return filtered_spectrum, foreground_model


def apply_windowed_delay_filter(
    spectrum: np.ndarray,
    weights: np.ndarray,
    freqs_mhz: np.ndarray,
    delay_cut_ns: float
) -> np.ndarray:
    """
    Applies a more realistic delay filter by using a smooth window function
    to suppress foreground-dominated modes in the delay domain.

    This method is more faithful to a real pipeline filter as it is less
    prone to ringing artifacts than a "brick-wall" cut.

    Args:
        spectrum: The 1D input data slice.
        weights: The weights array (0 for flagged channels).
        freqs_mhz: The frequency axis in MHz, used to calculate the cut.
        delay_cut_ns: The delay at which the filter's suppression begins, in ns.

    Returns:
        The real part of the filtered spectrum.
    """
    # 1. Calculate the total bandwidth in Hz
    bandwidth_hz = (np.max(freqs_mhz) - np.min(freqs_mhz)) * 1e6
    if bandwidth_hz == 0:
        return spectrum
        
    # 2. Convert the desired delay cut into an integer number of channels
    delay_cut_s = delay_cut_ns * 1e-9
    notch_width = int(np.round(delay_cut_s * bandwidth_hz))
    
    # 3. Create a smooth windowing function
    # A Tukey window is flat in the middle with tapered cosine edges.
    # We create a window that is the size of the notch on each side.
    num_modes_to_window = notch_width * 2
    if num_modes_to_window <= 0 or num_modes_to_window >= len(spectrum):
        return np.zeros_like(spectrum) # Return zero if the filter is too wide
        
    window = tukey(num_modes_to_window, alpha=1.0) # alpha=1.0 is a full cosine taper
    
    # The filter is a multiplication in the Fourier domain. 1 passes, 0 blocks.
    fft_filter = np.ones_like(spectrum, dtype=float)
    fft_filter[:notch_width] = window[:notch_width]
    fft_filter[-notch_width:] = window[notch_width:]
    
    # 4. Apply the filter
    # if weights = None then use uniform weights
    delay_spectrum = np.fft.fft(spectrum * weights)
    filtered_delay_spectrum = delay_spectrum * fft_filter
    filtered_spectrum = np.fft.ifft(filtered_delay_spectrum)
    
    return filtered_spectrum.real

    
# =============================================================================
# --- Component Functions: Sky and Weight Generation ---
# =============================================================================


def generate_gdsm_cube(
    num_pixels: int,
    freqs: np.ndarray
) -> np.ndarray:
    """
    Generates a full 3D data cube from the Global Sky Model (GDSM).

    This function iterates through each frequency channel, generates the GDSM
    for that frequency, projects it to a 2D grid, and stacks the results
    to create a physically realistic, frequency-dependent data cube.

    Args:
        num_pixels: The total number of spatial pixels in the output image.
        freqs: The array of channel center frequencies in MHz.

    Returns:
        A 3D numpy array representing the sky brightness cube (pixels, frequency).
    """
    grid_size = int(np.sqrt(num_pixels))
    if grid_size**2 != num_pixels:
        raise ValueError("GDSM model requires a perfect square num_pixels.")

    print("  - Initializing Global Sky Model...")
    # Initialize the GDSM model once
    gsm = GlobalSkyModel(freq_unit='MHz')
    
    # Create an empty data cube to store the results
    sky_cube = np.zeros((num_pixels, len(freqs)))

    # Iterate through each frequency channel and generate the sky slice
    for i, freq in enumerate(tqdm(freqs, desc="Generating GDSM slices")):
        # Generate the full sky map at this specific frequency
        healpix_map = gsm.generate(freq)
        
        # Project the HEALPix map to a 2D grid (e.g., Gnomonic projection)
        projected_map = hp.gnomview(
            healpix_map,
            rot=(120, 40, 0),  # A typical rotation to get a mix of galactic/extragalactic sky
            xsize=grid_size,
            ysize=grid_size,
            reso=1.5 * 60 / grid_size,
            return_projected_map=True,
            no_plot=True
        )
        # Flatten the 2D map and add it as a slice to our cube
        sky_cube[:, i] = projected_map.flatten()
        
    return sky_cube


def generate_instrument_weights(model_type: str, num_pixels: int) -> np.ndarray:
    """Dispatcher to generate the appropriate instrumental weights map."""
    print(f"  - Generating '{model_type}' noise weights.")
    if model_type == 'uniform':
        return np.ones(num_pixels)
    elif model_type == 'beam_weighted':
        grid_size = int(np.sqrt(num_pixels))
        if grid_size**2 != num_pixels: raise ValueError("Beam-weighted model needs a perfect square num_pixels.")
        return generate_sky_weights((grid_size, grid_size)).flatten()
    else:
        raise ValueError(f"Unknown noise_model: {model_type}")


# =============================================================================
# --- Simulation Orchestration ---
# =============================================================================


def generate_powerlaw_background(
    freqs: np.ndarray,
    reference_freq_mhz: float = 600.0,
    base_amplitude: float = 300.0,
    spectral_index: float = -2.5
) -> np.ndarray:
    """
    Generates a single, smooth power-law spectrum.

    This represents the diffuse, large-scale synchrotron emission of the galaxy.

    Args:
        freqs: The array of channel center frequencies in MHz.
        reference_freq_mhz: The frequency at which the base amplitude is defined.
        base_amplitude: The amplitude (e.g., in Kelvin) at the reference frequency.
        spectral_index: The exponent of the power law.

    Returns:
        A 1D numpy array representing the power-law spectrum.
    """
    print(f"  - Generating power-law background (index={spectral_index})...")
    # The model is T(f) = T_ref * (f / f_ref)^alpha
    return base_amplitude * (freqs / reference_freq_mhz)**spectral_index


def generate_point_sources_cube(
    num_pixels: int,
    freqs: np.ndarray,
    num_sources: int,
    amp_range: Tuple[float, float],
    spectral_index_range: Tuple[float, float],
    reference_freq_mhz: float = 600.0
) -> np.ndarray:
    """
    Generates a data cube containing only discrete point sources.

    Each source is given a random location, amplitude, and spectral index.

    Args:
        num_pixels: The total number of spatial pixels in the output image.
        freqs: The array of channel center frequencies in MHz.
        num_sources: The number of point sources to add to the cube.
        amp_range: A tuple (min, max) for the random source amplitudes.
        spectral_index_range: A tuple (min, max) for the random spectral indices.
        reference_freq_mhz: The frequency at which the source amplitudes are defined.

    Returns:
        A 2D numpy array (pixels, frequency) containing the point source emission.
    """
    print(f"  - Adding {num_sources} random point sources...")
    source_cube = np.zeros((num_pixels, len(freqs)))
    source_locations = np.random.choice(num_pixels, num_sources, replace=False)

    for pixel_idx in source_locations:
        # Draw random parameters for this source
        amplitude = np.random.uniform(*amp_range)
        spec_idx = np.random.uniform(*spectral_index_range)

        # Generate the source's power-law spectrum
        source_spectrum = amplitude * (freqs / reference_freq_mhz)**spec_idx
        source_cube[pixel_idx, :] += source_spectrum

    return source_cube

    
def _create_single_injection(
    pixel_idx: int,
    freqs: np.ndarray,
    vel_axis: np.ndarray,
    randomize_profile: bool,
    master_template_v: np.ndarray = None
) -> Tuple[np.ndarray, Dict]:
    """
    Generates a single, noiseless synthetic OHM signal profile.

    Args:
        pixel_idx: The pixel index where the signal will be placed.
        freqs: The frequency axis in MHz.
        vel_axis: The velocity axis in km/s for the intrinsic profile.
        randomize_profile: If True, a new random intrinsic profile is generated.
                           If False, `master_template_v` is used.
        master_template_v: A pre-generated intrinsic template. Required if
                           `randomize_profile` is False.

    Returns:
        - The full-band, noiseless profile of the injected signal.
        - The ground truth dictionary for this injection.
    """
    intrinsic_attributes = {} # Initialize empty dict
    
    if randomize_profile:
        # Now unpacks both the spectrum and the attributes dictionary
        intrinsic_template_v, intrinsic_attributes = otg.generate_intrinsic_maser_injection(vel_axis_kms=vel_axis)
    else:
        # Use the provided master template
        if master_template_v is None:
            raise ValueError("master_template_v must be provided when randomize_profile is False.")
        intrinsic_template_v = master_template_v
    
    # Randomize the redshift and amplitude for this injection
    z_inject = np.random.uniform(1.1, 3.1)
    amp_inject = np.abs(np.random.normal(loc=7, scale=2))
    
    # Redshift and resample the profile
    template, start_idx, end_idx = otg.process_to_native_resolution_and_target_z(
        intrinsic_template_v=intrinsic_template_v, vel_axis_kms=vel_axis,
        z=z_inject, native_freq_grid=freqs
    )
    if template is None or template.size == 0 or np.max(template) == 0:
        return None, None
        
    # Create and scale the full-band profile
    noiseless_profile = np.zeros_like(freqs)
    scaled_template = template * (amp_inject / np.max(template))
    noiseless_profile[start_idx:end_idx] = scaled_template
    
    # Create the base ground truth entry
    g_truth_entry = {
        'pixel_index': pixel_idx, 'z': z_inject, 'amp': amp_inject,
        'noiseless_profile': noiseless_profile
    }
    
    # Merge the intrinsic physical attributes into the ground truth dictionary
    g_truth_entry.update(intrinsic_attributes)
    
    return noiseless_profile, g_truth_entry


def generate_sky_image_cube(
    num_pixels: int,
    freqs: np.ndarray,
    noise_sigma_base: float = 0.5,
    num_injections: int = 100,
    sky_model: str = 'gdsm',
    randomize_injections: bool = False,
    injection_amp: Optional[float] = None
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a simulated sky data cube with a simple, uniform noise model.

    **UPDATED:** This version now returns clean, separate cubes for the
    foregrounds and the noise component for accurate diagnostics.

    Returns
    -------
    Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, np.ndarray]
        - data_cube: The final 2D cube (foregrounds + signals + noise).
        - ground_truth: A dictionary detailing all injected signals.
        - sky_weights: An array of uniform weights (all ones).
        - foreground_cube: The clean, noiseless foreground component.
        - noise_cube: The clean Gaussian noise component.
    """
    print(f"Generating simulation cube with {num_pixels} pixels...")
    num_freqs = len(freqs)

    # --- Step 1: Generate the Foreground Sky Model ---
    print(f"  - Generating '{sky_model}' foreground model...")
    if sky_model == 'gdsm':
        foreground_cube = generate_gdsm_cube(num_pixels, freqs)
    else: # 'blank' or other models
        foreground_cube = np.zeros((num_pixels, num_freqs))

    # --- Step 2: Generate Simple, Uniform Instrumental Noise ---
    print(f"  - Adding simple uniform Gaussian noise...")
    noise_cube = np.random.normal(0, noise_sigma_base, (num_pixels, num_freqs))
    
    # Combine the two to create the base data cube
    data_cube = foreground_cube + noise_cube
    
    # The sky_weights are simple and uniform.
    sky_weights = np.ones((num_pixels, num_freqs))

    # --- Step 3: Inject Synthetic OHM Signals ---
    print(f"  - Injecting {num_injections} synthetic OHM signals...")
    vel_axis = np.linspace(-1200, 1200, 4096)
    
    if randomize_injections:
        master_template = None
    else:
        master_template = otg.generate_optimal_template(vel_axis_kms=vel_axis, N_population=5000, verbose=False)

    ground_truth = {'injections': []}
    injection_indices = np.random.choice(num_pixels, num_injections, replace=False)

    for pixel_idx in tqdm(injection_indices, desc="Injecting Signals"):
        noiseless_profile, g_truth_entry = _create_single_injection(
            pixel_idx=pixel_idx,
            freqs=freqs,
            vel_axis=vel_axis,
            randomize_profile=randomize_injections,
            master_template_v=master_template
        )
        
        if noiseless_profile is not None:
            if injection_amp is not None:
                old_amp = g_truth_entry['amp']
                scale_factor = injection_amp / old_amp
                noiseless_profile *= scale_factor
                g_truth_entry['amp'] = injection_amp

            ground_truth['injections'].append(g_truth_entry)
            # Add the signal to the final data cube
            data_cube[pixel_idx, :] += noiseless_profile
            
    # Return all components, including the new separate cubes
    return data_cube, ground_truth, sky_weights, foreground_cube, noise_cube

# =============================================================================
# --- Search and Filtering Algorithms ---
# =============================================================================


def run_matched_filter_direct(
    spectrum: np.ndarray,
    template: np.ndarray,
    weights: np.ndarray,
    noise_sigma: float
) -> float:
    """
    Performs a matched filter operation between a spectrum and a template.

    This corrected version properly applies the weights (e.g., an RFI mask)
    to both the spectrum and the template.
    """
    # Ensure weights are a boolean or float array that can be multiplied
    weights = weights.astype(float)

    # Apply the weights to both the data and the template. This is the crucial step.
    # It effectively sets the values in RFI-flagged channels to zero.
    weighted_spectrum = spectrum * weights
    weighted_template = template * weights

    # The numerator is the dot product of the weighted signals
    numerator = np.dot(weighted_spectrum, weighted_template)

    # The denominator must also be calculated from the weighted template
    # to correctly normalize the SNR.
    denominator = noise_sigma * np.sqrt(np.dot(weighted_template, weighted_template))

    # Prevent division by zero if the template is completely masked by RFI
    if denominator == 0:
        return 0.0

    return numerator / denominator


def run_matched_filter_cube(
    data_cube: np.ndarray,
    template_bank_full: List[Dict[str, Any]],
    weights: np.ndarray,
    noise_spectrum: np.ndarray = None,
) -> np.ndarray:
    """
    Applies a matched filter to each frequency bin of a data cube to generate a full 3D SNR cube.

    Args:
        data_cube: The 3D numpy array of sky data (pixel, frequency).
        templates: A list of template arrays, one for each frequency channel.
        weights: A 1D array of weights for the frequency channels.
        noise_sigma: The standard deviation of the noise.

    Returns:
        A 3D numpy array representing the SNR cube (pixel, frequency).
    """
    num_pixels, num_freqs = data_cube.shape
    snr_cube = np.zeros_like(data_cube)

    # 1. Use a robust MAD-based noise estimator if not provided
    if noise_spectrum is None:
        print("  - Estimating noise with robust MAD estimator...")
        # Median Absolute Deviation is more robust to outliers than std dev
        median_abs_dev = np.nanmedian(np.abs(data_cube - np.nanmedian(data_cube, axis=0)), axis=0)
        noise_spectrum = median_abs_dev * 1.4826 # Conversion factor for equivalence to std dev for Gaussian noise

    for i in tqdm(range(num_pixels), desc="Applying Matched Filter to each Frequency Bin"):
        spectrum = data_cube[i, :]
        for j in range(num_freqs):
            noise_sigma = noise_spectrum[j] # noise estimate is per channel
            template = template_bank_full[j]['prof']
            template = template/template.max()
            snr_cube[i, j] = run_matched_filter_direct(spectrum, template, weights, noise_sigma)

    return snr_cube, noise_spectrum


def run_threshold_search(filtered_spectrum: np.ndarray, noise_sigma_per_channel: np.ndarray) -> float:
    """
    Finds the peak significance of a signal in a single channel.

    This function serves as a simple, baseline search method. It calculates the
    significance (or signal-to-noise ratio) for each individual channel by
    dividing its amplitude by the corresponding noise standard deviation. It
    then returns the single highest significance value found across the entire
    spectrum.

    This is useful for comparing against a matched filter, which should achieve
    a higher SNR by combining signal across multiple channels.

    Parameters
    ----------
    filtered_spectrum : np.ndarray
        A 1D numpy array of the spectrum data, assumed to have had its
        continuum or foregrounds removed.
    noise_sigma_per_channel : np.ndarray
        A 1D numpy array of the same shape as `filtered_spectrum`, where each
        element is the standard deviation (sigma) of the noise for the
        corresponding channel.

    Returns
    -------
    float
        The maximum single-channel significance (SNR) found in the spectrum.
    """
    # Create an array to store the significance value for each channel.
    significance_spectrum = np.zeros_like(filtered_spectrum)

    # Create a boolean mask to identify channels where noise is defined.
    # This is a safety check to prevent division by zero errors.
    valid_indices = noise_sigma_per_channel > 0

    # For all valid channels, calculate the significance by dividing the
    # channel's amplitude by its noise sigma.
    significance_spectrum[valid_indices] = (
        filtered_spectrum[valid_indices] / noise_sigma_per_channel[valid_indices]
    )

    # Find and return the single highest significance value from the results.
    return np.max(significance_spectrum)


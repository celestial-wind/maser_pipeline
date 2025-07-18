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

import numpy as np
from tqdm.auto import tqdm
from typing import Dict, Tuple, List, Any
from scipy.signal import correlate
from scipy.signal.windows import tukey

# External library dependencies
import healpy as hp
from pygdsm import GlobalSkyModel
from uvtools import dspec

# Local module dependencies
import ohm_template_generator as otg


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
    frequencies: np.ndarray,
    fit_mode: str = 'global',
    local_fit_padding: int = 10
) -> np.ndarray:
    """
    Infills RFI-flagged regions in a spectrum using a power-law fit.

    This function identifies flagged regions (where weight is 0), fits a
    power-law model to the surrounding valid data, and uses this model to
    interpolate values for the flagged channels. It then adds synthetic noise
    based on the spectrum's characteristics.

    Parameters
    ----------
    spectrum : np.ndarray
        The 1D array of the spectrum data, potentially containing RFI.
    weights : np.ndarray
        A 1D array of weights. Channels with a weight of 0 are considered
        flagged for RFI and will be infilled.
    frequencies : np.ndarray
        The corresponding frequency axis for the spectrum.
    fit_mode : str, optional
        The method for fitting the power law. Can be 'global' or 'local'.
        - 'global': Fits a single power law to all valid data in the spectrum.
          Faster but less accurate if the spectral index varies.
        - 'local': Fits a separate power law for each RFI zone using nearby
          valid data. More accurate but slower.
        Defaults to 'local'.
    local_fit_padding : int, optional
        The number of valid channels on each side of an RFI zone to use for
        the fit when `fit_mode` is 'local'. Defaults to 10.

    Returns
    -------
    np.ndarray
        A new spectrum array with the RFI-flagged zones infilled.
    """
    # --- 1. Setup and Input Validation ---
    if fit_mode not in ['global', 'local']:
        raise ValueError("fit_mode must be either 'global' or 'local'")

    # Create a copy of the spectrum to modify and return
    infilled_spectrum = np.copy(spectrum)

    # Identify valid (unflagged) and invalid (flagged) data points
    valid_indices = weights > 0
    flagged_indices = ~valid_indices

    # If there's nothing to infill, return the original spectrum
    if not np.any(flagged_indices):
        return infilled_spectrum

    # --- 2. Fit Power-Law Model ---
    # A power law y = A * x^k becomes linear in log-log space:
    # log(y) = k * log(x) + log(A). We fit this line.
    power_law_model = np.zeros_like(spectrum)

    if fit_mode == 'global':
        # Use all valid data for a single fit
        log_freq = np.log10(frequencies[valid_indices])
        log_spec = np.log10(spectrum[valid_indices])
        # Fit a line (degree 1 polynomial) to the log-log data
        k, log_A = np.polyfit(log_freq, log_spec, 1)
        # Generate the model for all frequencies
        power_law_model = (10**log_A) * (frequencies**k)

    elif fit_mode == 'local':
        # Find contiguous blocks of flagged channels
        labeled_zones, num_zones = label(flagged_indices)
        for i in range(1, num_zones + 1):
            zone_indices = np.where(labeled_zones == i)[0]
            start_index, end_index = zone_indices[0], zone_indices[-1]

            # Select padding data on both sides of the RFI zone
            left_indices = np.where(valid_indices & (np.arange(len(spectrum)) < start_index))[0]
            right_indices = np.where(valid_indices & (np.arange(len(spectrum)) > end_index))[0]

            # Ensure we have enough points for a stable fit
            if len(left_indices) < 2 or len(right_indices) < 2:
                # Fallback to a global fit if a local one is not possible
                k, log_A = np.polyfit(np.log10(frequencies[valid_indices]), np.log10(spectrum[valid_indices]), 1)
            else:
                fit_indices = np.concatenate([left_indices[-local_fit_padding:], right_indices[:local_fit_padding]])
                log_freq = np.log10(frequencies[fit_indices])
                log_spec = np.log10(spectrum[fit_indices])
                k, log_A = np.polyfit(log_freq, log_spec, 1)

            # Generate model values just for this flagged zone
            power_law_model[zone_indices] = (10**log_A) * (frequencies[zone_indices]**k)

    # --- 3. Infill and Add Noise ---
    # First, infill the flagged regions with the power-law model
    infilled_spectrum[flagged_indices] = power_law_model[flagged_indices]

    # Measure the noise from the valid regions of the original spectrum
    # The noise is the standard deviation of the data after subtracting the model
    residual = spectrum[valid_indices] - power_law_model[valid_indices]
    # Ensure we don't include NaNs or zeros in the noise calculation
    noise_sigma = np.std(residual[np.isfinite(residual)])

    # Generate synthetic noise for the flagged channels
    synthetic_noise = np.random.normal(0, noise_sigma, size=np.sum(flagged_indices))

    # Add the synthetic noise to the infilled regions
    infilled_spectrum[flagged_indices] += synthetic_noise

    return infilled_spectrum


def apply_dayneu_filter(spectrum, frequencies_mhz, delay_cutoff_ns, weights=None):
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
        filter_dims=1
    )

    return filtered_spectrum, foreground_model


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


def generate_sky_image_cube(
    num_pixels: int, 
    freqs: np.ndarray, 
    noise_sigma_base: float = 0.5,
    num_injections: int = 100, 
    sky_model: str = 'gdsm',
    noise_model: str = 'uniform',
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """
    Generates a simulated sky data cube with selectable physical models.
    """
    print(f"Generating simulation cube with {num_pixels} pixels...")

    # --- New, cleaner sky model generation ---
    print(f"  - Generating '{sky_model}' sky model.")
    if sky_model == 'blank':
        data_cube = np.zeros((num_pixels, len(freqs)))
    elif sky_model == 'gdsm':
        # Call the new, more powerful function
        data_cube = generate_gdsm_cube(num_pixels, freqs)
    else:
        raise ValueError(f"Unknown sky_model: {sky_model}")
    # ----------------------------------------

    # Get instrumental weights based on selected noise model
    sky_weights = generate_instrument_weights(noise_model, num_pixels)
    
    # Inject synthetic OHM signals
    ground_truth = {'injections': []}
    vel_axis = np.linspace(-1200, 1200, 4096)
    injection_indices = np.random.choice(num_pixels, num_injections, replace=False)
    # Pre-generate the master intrinsic template once for efficiency
    intrinsic_template_v = otg.generate_optimal_template(vel_axis_kms=vel_axis, N_population=5000, verbose=False)

    for pixel_idx in tqdm(injection_indices, desc="Injecting Signals"):
        z_inject = np.random.uniform(1.1, 3.1)
        amp_inject = np.abs(np.random.normal(loc=7, scale=2))
        
        # Transform the master template for this specific injection
        template, start_idx, end_idx = otg.process_to_native_resolution_and_target_z(
            intrinsic_template_v=intrinsic_template_v, vel_axis_kms=vel_axis,
            z=z_inject, native_freq_grid=freqs
        )
        if template.size == 0 or np.max(template) == 0: continue
        
        noiseless_profile = np.zeros_like(freqs)
        scaled_template = template * (amp_inject / np.max(template))
        noiseless_profile[start_idx:end_idx] = scaled_template
        
        g_truth_entry = {'pixel_index': pixel_idx, 'z': z_inject, 'amp': amp_inject, 'noiseless_profile': noiseless_profile}
        ground_truth['injections'].append(g_truth_entry)
        
        # Add the signal directly into the GDSM cube
        data_cube[pixel_idx, :] += noiseless_profile

    # Add instrumental noise, scaled by the weights
    print("\n  - Adding instrumental noise...")
    for i in tqdm(range(num_pixels), desc="Adding Noise"):
        pixel_noise_sigma = noise_sigma_base / sky_weights[i]
        noise = np.random.normal(0, pixel_noise_sigma, len(freqs))
        data_cube[i, :] += noise
        
    return data_cube, ground_truth, sky_weights

    
# =============================================================================
# --- Search and Filtering Algorithms ---
# =============================================================================


def run_matched_filter_direct(
    spectrum: np.ndarray,
    template: np.ndarray,
    weights: np.ndarray,
    noise_sigma: float,
    frequencies_mhz: np.ndarray,
    filter_template: bool = False,
    delay_cutoff_ns: float = 200.0
) -> float:
    """
    Calculates the SNR of a signal at a specific, known alignment using a
    weighted matched filter.

    This function performs a direct summation "convolution" (dot product) at a
    single lag, which is suitable for data with masked or flagged RFI zones.

    Parameters
    ----------
    spectrum : np.ndarray
        The 1D array of the observed spectrum containing the signal and noise.
    template : np.ndarray
        The 1D array representing the ideal, noise-free signal template.
    weights : np.ndarray
        A 1D array of weights corresponding to the spectrum. RFI-flagged
        channels should have a weight of 0.
    noise_sigma : float
        The standard deviation (sigma) of the noise in the spectrum.
    frequencies_mhz : np.ndarray
        The frequency axis for the spectrum in MHz. Required for filtering.
    filter_template : bool, optional
        If True, the template will be filtered with the Dayenu delay filter
        before matching. This is the recommended approach if the spectrum
        has been filtered. Defaults to False, which uses a simple mean
        subtraction as an approximation.
    delay_cutoff_ns : float, optional
        The delay cutoff in nanoseconds to use for the Dayenu filter if
        `filter_template` is True. Defaults to 200.0.

    Returns
    -------
    float
        The calculated signal-to-noise ratio (SNR) of the signal.
    """
    if filter_template:
        # If toggled, apply the proper Dayenu filter to the template.
        # The first return value is the filtered template.
        template_norm, _ = apply_dayneu_filter(
            spectrum=template,
            frequencies_mhz=frequencies_mhz,
            delay_cutoff_ns=delay_cutoff_ns,
            weights=weights
        )
    else:
        # Original behavior: approximate filtering by subtracting the mean.
        template_norm = template - np.mean(template)

    # Calculate the weighted energy of the normalized template
    template_energy = np.sum(template_norm**2 * weights)

    # Avoid division by zero if template is flat or noise is zero
    if template_energy == 0 or noise_sigma <= 0:
        return 0.0

    # Calculate the matched filter score (dot product of spectrum and template)
    score = np.sum(spectrum * template_norm * weights)

    # Return the SNR: the score normalized by template energy and noise level
    return score / (np.sqrt(template_energy) * noise_sigma)


def run_matched_filter_fft(
    spectrum: np.ndarray,
    template: np.ndarray,
    noise_sigma: float,
    frequencies_mhz: np.ndarray,
    filter_template: bool = False,
    delay_cutoff_ns: float = 200.0,
    align_output: bool = True
) -> np.ndarray:
    """
    Performs a matched filter by correlating a template across a
    spectrum, returning a full SNR spectrum.

    This function is optimized for data without significant RFI flags,
    as it uses `np.correlate` which is efficient for full arrays.

    Parameters
    ----------
    spectrum : np.ndarray
        The 1D array of the observed spectrum to be searched.
    template : np.ndarray
        The 1D array representing the ideal, noise-free signal template.
        Should be the same size as `spectrum`.
    noise_sigma : float
        The standard deviation (sigma) of the noise in the spectrum.
    frequencies_mhz : np.ndarray
        The frequency axis for the spectrum in MHz. Required if filtering
        the template.
    filter_template : bool, optional
        If True, the template will be filtered with the Dayenu delay filter
        before matching. This is recommended if the spectrum has been
        filtered. Defaults to False.
    delay_cutoff_ns : float, optional
        The delay cutoff in nanoseconds to use for the Dayenu filter if
        `filter_template` is True. Defaults to 200.0.
    align_output : bool, optional
        If True, corrects the alignment of the output SNR spectrum to ensure
        the peak location corresponds to the signal's true location.
        Defaults to True.

    Returns
    -------
    np.ndarray
        A 1D array of the same size as `spectrum`, where each value
        represents the SNR of a potential signal at that channel.
    """
    # --- 1. Prepare the Template ---
    if filter_template:
        # Apply the proper Dayenu filter to the template.
        template_norm, _ = apply_dayneu_filter(
            spectrum=template,
            frequencies_mhz=frequencies_mhz,
            delay_cutoff_ns=delay_cutoff_ns
        )
    else:
        # Default behavior: approximate filtering by subtracting the mean.
        template_norm = template - np.mean(template)

    # --- 2. Calculate Correlation and SNR ---
    template_energy = np.sum(template_norm**2)
    if template_energy == 0 or noise_sigma <= 0:
        return np.zeros_like(spectrum)

    # Correlate the spectrum with the normalized template
    correlation = np.correlate(spectrum, template_norm, mode='same')
    snr_unaligned = correlation / (np.sqrt(template_energy) * noise_sigma)

    # --- 3. Align the Output ---
    if align_output:
        # Find the peak of the unaligned correlation output
        correlation_peak_index = np.argmax(snr_unaligned)
        # Find the peak of the prepared template
        template_peak_index = np.argmax(np.abs(template_norm))

        # Calculate the shift and roll the array for proper alignment
        shift_amount = correlation_peak_index + template_peak_index
        snr_spectrum = np.roll(snr_unaligned, shift_amount)
    else:
        # Return the unaligned SNR spectrum if alignment is turned off
        snr_spectrum = snr_unaligned

    return snr_spectrum

import numpy as np


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


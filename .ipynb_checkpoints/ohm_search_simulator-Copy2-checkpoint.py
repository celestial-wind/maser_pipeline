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
from typing import Dict, Tuple, List, Any, Optional, Union
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
    noise_model: str = 'uniform',
    return_foregrounds_only: bool = False,
    randomize_injections: bool = False,
    injection_amp: Optional[float] = None,
    pl_spec_index: float = -2.5,
    pl_base_amp_k: float = 300.0,
    num_point_sources: int = 50,
    ps_amp_range_k: Tuple[float, float] = (10.0, 100.0),
    ps_spec_idx_range: Tuple[float, float] = (-2.8, -2.2)
) -> Tuple[np.ndarray, Dict, np.ndarray, Optional[np.ndarray]]:
    """
    Generates a simulated sky data cube with selectable physical models.

    This function builds a data cube by layering three components:
    1. A smooth-spectrum foreground model (e.g., GDSM).
    2. A set of injected, synthetic OHM signals.
    3. Instrumental noise with specified properties.

    Parameters
    ----------
    num_pixels : int
        The number of independent pixels (lines of sight) in the data cube.
    freqs : np.ndarray
        The 1D array of frequency channels in MHz.
    noise_sigma_base : float, optional
        The base standard deviation of the noise for a uniformly weighted pixel.
        Defaults to 0.5.
    num_injections : int, optional
        The number of synthetic OHM signals to inject into the cube.
        Defaults to 100.
    sky_model : str, optional
        The foreground model to use ('gdsm', 'blank', or 'powerlaw_sources').
        Defaults to 'gdsm'.
    noise_model : str, optional
        The noise weighting model to use ('uniform' or other defined models).
        Defaults to 'uniform'.
    return_foregrounds_only : bool, optional
        If True, an additional data cube containing only the pure foreground
        component will be returned before signals and noise are added.
        Defaults to False.
    pl_spec_index (float, optional): The spectral index alpha for the smooth,
        diffuse power-law background component, where brightness T is
        proportional to f^alpha. A typical value for galactic synchrotron
        emission is around -2.5. Defaults to -2.5.
    pl_base_amp_k (float, optional): The base amplitude of the diffuse
        background in Kelvin, defined at a reference frequency of 600 MHz.
        Defaults to 300.0.
    num_point_sources (int, optional): The total number of discrete,
        unresolved point sources to inject into the foreground model.
        Defaults to 50.
    ps_amp_range_k (Tuple[float, float], optional): The range (min, max)
        from which to draw a random amplitude in Kelvin for each point
        source (at the 600 MHz reference frequency).
        Defaults to (10.0, 100.0).
    ps_spec_idx_range (Tuple[float, float], optional): The range (min, max)
        from which to draw a random spectral index for each individual
        point source. This allows for source-to-source variation in their
        spectra. Defaults to (-2.8, -2.2).

    Returns
    -------
    Tuple[np.ndarray, Dict, np.ndarray, Optional[np.ndarray]]
        - data_cube (np.ndarray): The final 2D data cube containing
          (foregrounds + signals + noise).
        - ground_truth (Dict): A dictionary detailing all injected signals.
        - sky_weights (np.ndarray): The per-pixel instrumental weights.
        - foreground_cube (np.ndarray or None): If `return_foregrounds_only`
          is True, this is the 2D cube with only the foregrounds. Otherwise, None.
    """
    print(f"Generating simulation cube with {num_pixels} pixels...")

    # --- Step 1: Generate the Foreground Sky Model ---
    print(f"  - Generating '{sky_model}' sky model...")
    if sky_model == 'blank':
        data_cube = np.zeros((num_pixels, len(freqs)))
    elif sky_model == 'gdsm':
        data_cube = generate_gdsm_cube(num_pixels, freqs)
    elif sky_model == 'powerlaw_sources':
        # 1. Create the smooth, uniform background component
        background_spec = generate_powerlaw_background(
            freqs, base_amplitude=pl_base_amp_k, spectral_index=pl_spec_index
        )
        # Tile it to match the cube shape
        data_cube = np.tile(background_spec, (num_pixels, 1))

        # 2. Create and add the point source component
        point_source_cube = generate_point_sources_cube(
            num_pixels, freqs,
            num_sources=num_point_sources,
            amp_range=ps_amp_range_k,
            spectral_index_range=ps_spec_idx_range
        )
        data_cube += point_source_cube
    else:
        raise ValueError(f"Unknown sky_model: {sky_model}")

    # If requested, store a clean copy of the foregrounds now,
    # before adding any other components.
    foreground_cube = np.copy(data_cube) if return_foregrounds_only else None

    # --- Step 2: Inject Synthetic OHM Signals ---
    # This ensures it's always available for both injection modes.
    vel_axis = np.linspace(-1200, 1200, 4096)
    
    if randomize_injections:
        print(f"  - Injecting {num_injections} RANDOMIZED synthetic OHM signals...")
        master_template = None
    else:
        print(f"  - Injecting {num_injections} IDENTICAL synthetic OHM signals...")
        # Pre-generate the master intrinsic template once for efficiency
        master_template = otg.generate_optimal_template(vel_axis_kms=vel_axis, N_population=5000, verbose=False)

    ground_truth = {'injections': []}
    injection_indices = np.random.choice(num_pixels, num_injections, replace=False)
    
    for pixel_idx in tqdm(injection_indices, desc="Injecting Signals"):
        # The injection logic is now neatly contained in the helper function
        noiseless_profile, g_truth_entry = _create_single_injection(
            pixel_idx=pixel_idx,
            freqs=freqs,
            vel_axis=vel_axis,
            randomize_profile=randomize_injections,
            master_template_v=master_template
        )
        
        if noiseless_profile is not None:
            # If a fixed amplitude is given, override the random one
            if injection_amp is not None:
                # Find the old amplitude to scale correctly
                old_amp = g_truth_entry['amp']
                scale_factor = injection_amp / old_amp
                noiseless_profile *= scale_factor
                g_truth_entry['amp'] = injection_amp

            ground_truth['injections'].append(g_truth_entry)
            data_cube[pixel_idx, :] += noiseless_profile

    # --- Step 3: Add Instrumental Noise ---
    print(f"\n  - Adding instrumental noise based on '{noise_model}' model...")
    sky_weights = generate_instrument_weights(noise_model, num_pixels)
    for i in tqdm(range(num_pixels), desc="Adding Noise"):
        # Scale noise by the per-pixel instrumental weight
        pixel_noise_sigma = noise_sigma_base / sky_weights[i]
        noise = np.random.normal(0, pixel_noise_sigma, len(freqs))
        data_cube[i, :] += noise
        
    return data_cube, ground_truth, sky_weights, foreground_cube

    
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


# def run_whitened_correlation(
#     data_cube: np.ndarray,
#     template: np.ndarray,
#     noise_spectrum: np.ndarray = None
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Performs a whitened matched filter correlation on a data cube using a
#     single template, returning a full SNR spectrum for each pixel.
#     This function is the recommended standard for obtaining a sensitive,
#     linear, and localized filter response.
#     Args:
#         data_cube: The 2D data cube (pixels x frequencies) to be searched.
#         template: A single 1D array for the signal template.
#         noise_spectrum: A pre-calculated 1D array of the per-channel noise
#                         standard deviation. If None, it will be estimated.
#     Returns:
#         A tuple containing:
#         - snr_cube: The 2D cube of whitened SNR spectra.
#         - noise_spectrum: The 1D noise spectrum used for whitening.
#     """
#     print("--- Running Whitened Correlation Search ---")
    
#     # 1. Estimate noise if not provided
#     if noise_spectrum is None:
#         print("  - Estimating noise spectrum from the data cube...")
#         # Use robust noise estimation to avoid outliers
#         noise_spectrum = np.nanmedian(np.abs(data_cube - np.nanmedian(data_cube, axis=0)), axis=0) * 1.4826
    
#     # Ensure noise has a reasonable floor to prevent division by very small numbers
#     # Set floor relative to the median noise level
#     median_noise = np.nanmedian(noise_spectrum)
#     noise_floor = max(1e-6 * median_noise, 1e-12)  # Adaptive floor
#     noise_spectrum = np.where(noise_spectrum <= noise_floor, noise_floor, noise_spectrum)
    
#     # Debug: Print noise statistics
#     print(f"  - Noise spectrum range: {np.min(noise_spectrum):.2e} to {np.max(noise_spectrum):.2e}")
#     print(f"  - Median noise: {median_noise:.2e}")
    
#     # 2. Whiten the template and calculate normalization factor
#     whitened_template = template / noise_spectrum
#     template_norm_squared = np.sum(whitened_template**2)
    
#     if template_norm_squared == 0:
#         print("Warning: Template has zero energy after whitening. Returning zeros.")
#         return np.zeros_like(data_cube), noise_spectrum
    
#     print(f"  - Template norm squared: {template_norm_squared:.2e}")
    
#     # For proper SNR normalization in matched filter
#     snr_normalization = np.sqrt(template_norm_squared)
    
#     # Prepare template for matched filtering (time-reversed for correlation)
#     matched_filter_template = whitened_template[::-1]
    
#     # 3. Prepare output array and process each pixel
#     snr_cube = np.zeros_like(data_cube)
#     max_snr_seen = 0
    
#     for i in tqdm(range(data_cube.shape[0]), desc="Processing Pixels"):
#         spectrum = data_cube[i, :]
        
#         # Handle NaN values in the spectrum
#         if np.any(np.isnan(spectrum)):
#             snr_cube[i, :] = np.nan
#             continue
            
#         # Whiten the data for the current pixel
#         whitened_spectrum = spectrum / noise_spectrum
        
#         # Perform the matched filter correlation (with time-reversed template)
#         matched_filter_output = np.correlate(whitened_spectrum, matched_filter_template, mode='same')
        
#         # Normalize to get the final SNR spectrum
#         snr_spectrum = matched_filter_output / snr_normalization
        
#         # Track maximum SNR for debugging
#         pixel_max_snr = np.max(np.abs(snr_spectrum))
#         if pixel_max_snr > max_snr_seen:
#             max_snr_seen = pixel_max_snr
            
#         # Optional: Cap extremely large values that might indicate numerical issues
#         snr_spectrum = np.clip(snr_spectrum, -1000, 1000)
        
#         #snr_cube[i, :] = snr_spectrum
#         snr_cube[i, :] = np.where(snr_spectrum > 0, snr_spectrum, 0)

        
#     print(f"  - Maximum |SNR| encountered: {max_snr_seen:.2f}")
    
#     return snr_cube, noise_spectrum
    

def run_whitened_correlation(
    data_cube: np.ndarray,
    template: np.ndarray,
    noise_spectrum: np.ndarray = None,
    enforce_positivity: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a whitened matched filter correlation on a data cube, with an
    option to make the output compatible with threshold-based search algorithms.

    This is the final recommended function.

    Args:
        data_cube: The 2D data cube (pixels x frequencies) to be searched.
        template: A single 1D array for the signal template.
        noise_spectrum: A pre-calculated 1D noise spectrum. If None, it will
                        be estimated using a robust MAD estimator.
        enforce_positivity: If True, sets all negative SNR values in the
                            output cube to zero. This is recommended for use
                            with the find_candidates_3d_dbscan algorithm.

    Returns:
        A tuple containing the SNR cube and the noise spectrum.
    """
    print("--- Running Final Whitened Correlation Search ---")

    # 1. Use a robust MAD-based noise estimator if not provided
    if noise_spectrum is None:
        print("  - Estimating noise with robust MAD estimator...")
        # Median Absolute Deviation is more robust to outliers than std dev
        median_abs_dev = np.nanmedian(np.abs(data_cube - np.nanmedian(data_cube, axis=0)), axis=0)
        noise_spectrum = median_abs_dev * 1.4826 # Conversion factor for equivalence to std dev for Gaussian noise
    
    noise_floor = 1e-9
    noise_spectrum[np.isnan(noise_spectrum) | (noise_spectrum <= noise_floor)] = noise_floor

    # 2. Whiten the template and calculate its normalization factor
    whitened_template = template / noise_spectrum
    template_norm = np.sqrt(np.sum(whitened_template**2))
    
    if template_norm == 0:
        return np.zeros_like(data_cube), noise_spectrum

    # 3. Process each pixel
    snr_cube = np.zeros_like(data_cube)
    for i in tqdm(range(data_cube.shape[0]), desc="Processing Pixels"):
        spectrum = data_cube[i, :]
        whitened_spectrum = spectrum / noise_spectrum
        
        correlation = np.correlate(whitened_spectrum, whitened_template, mode='same')
        
        snr_spectrum = correlation / template_norm
        snr_cube[i, :] = snr_spectrum
        
    # 4. Enforce positivity if requested
    if enforce_positivity:
        print("  - Enforcing positivity on final SNR cube.")
        snr_cube[snr_cube < 0] = 0
        
    return snr_cube, noise_spectrum

# def run_whitened_correlation_aligned(
#     data_cube: np.ndarray,
#     template: np.ndarray,
#     noise_spectrum: np.ndarray = None
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Performs a whitened matched filter with proper alignment preservation.
#     The output SNR cube will have peaks at the correct positions where 
#     template matches occur in the original data.
    
#     Args:
#         data_cube: The 2D data cube (pixels x frequencies) to be searched.
#         template: A single 1D array for the signal template.
#         noise_spectrum: A pre-calculated 1D array of the per-channel noise
#                         standard deviation. If None, it will be estimated.
#     Returns:
#         A tuple containing:
#         - snr_cube: The 2D cube of whitened SNR spectra with proper alignment.
#         - noise_spectrum: The 1D noise spectrum used for whitening.
#     """
#     print("--- Running Aligned Whitened Correlation Search ---")
    
#     # 1. Estimate noise if not provided
#     if noise_spectrum is None:
#         print("  - Estimating noise spectrum from the data cube...")
#         noise_spectrum = np.nanmedian(np.abs(data_cube - np.nanmedian(data_cube, axis=0)), axis=0) * 1.4826
    
#     # Ensure noise has a reasonable floor
#     median_noise = np.nanmedian(noise_spectrum)
#     noise_floor = max(1e-6 * median_noise, 1e-12)
#     noise_spectrum = np.where(noise_spectrum <= noise_floor, noise_floor, noise_spectrum)
    
#     # 2. Whiten the template and calculate normalization factor
#     whitened_template = template / noise_spectrum
#     template_norm_squared = np.sum(whitened_template**2)
    
#     if template_norm_squared == 0:
#         print("Warning: Template has zero energy after whitening. Returning zeros.")
#         return np.zeros_like(data_cube), noise_spectrum
    
#     snr_normalization = np.sqrt(template_norm_squared)
    
#     # 3. Choose alignment method
#     n_data = data_cube.shape[1]
#     n_template = len(template)
    
#     print(f"  - Data length: {n_data}, Template length: {n_template}")
    
#     # Method 1: FFT-based with proper alignment (recommended)
#     snr_cube = _fft_matched_filter_aligned(data_cube, whitened_template, snr_normalization, noise_spectrum)
    
#     # Alternative: Correlation-based with offset correction
#     # snr_cube = _correlation_matched_filter_aligned(data_cube, whitened_template, snr_normalization)
    
#     return snr_cube, noise_spectrum


# def _fft_matched_filter_aligned(data_cube, whitened_template, snr_normalization, noise_spectrum):
#     """
#     FFT-based matched filter with proper alignment preservation.
#     """
#     n_pixels, n_data = data_cube.shape
#     n_template = len(whitened_template)
    
#     # Zero-pad template to match data length and time-reverse
#     padded_template = np.zeros(n_data)
#     padded_template[:n_template] = whitened_template[::-1]
    
#     # FFT of the matched filter template
#     template_fft = np.fft.fft(padded_template)
    
#     snr_cube = np.zeros_like(data_cube)
    
#     for i in tqdm(range(n_pixels), desc="Processing Pixels (FFT)"):
#         spectrum = data_cube[i, :]
        
#         if np.any(np.isnan(spectrum)):
#             snr_cube[i, :] = np.nan
#             continue
            
#         # Whiten the data
#         whitened_spectrum = spectrum / noise_spectrum
        
#         # Matched filter in frequency domain
#         data_fft = np.fft.fft(whitened_spectrum)
#         matched_output = np.fft.ifft(data_fft * np.conj(template_fft)).real
        
#         # The matched filter output is already properly aligned
#         # Peak at index i means template centered at position i in original data
#         snr_spectrum = matched_output / snr_normalization
        
#         snr_cube[i, :] = snr_spectrum
        
#     return snr_cube
    

def run_whitened_correlation_aligned(
    data_cube: np.ndarray,
    template: np.ndarray,
    noise_spectrum: np.ndarray = None,
    enforce_positivity: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a whitened matched filter and manually aligns the output
    so the peak of the SNR spectrum is shifted to a consistent location.

    Args:
        data_cube: The 2D data cube (pixels x frequencies) to be searched.
        template: A single 1D array for the signal template.
        noise_spectrum: A pre-calculated 1D noise spectrum. If None, it will
                        be estimated using a robust MAD estimator.

    Returns:
        A tuple containing the aligned SNR cube and the noise spectrum.
    """
    print("--- Running Aligned Whitened Correlation Search ---")

    # 1. Estimate noise if not provided
    if noise_spectrum is None:
        print("  - Estimating noise with robust MAD estimator...")
        median_abs_dev = np.nanmedian(np.abs(data_cube - np.nanmedian(data_cube, axis=0)), axis=0)
        noise_spectrum = median_abs_dev * 1.4826
    
    noise_floor = 1e-9
    noise_spectrum[np.isnan(noise_spectrum) | (noise_spectrum <= noise_floor)] = noise_floor

    # 2. Whiten the template and calculate its norm
    whitened_template = template / noise_spectrum
    template_norm = np.sqrt(np.sum(whitened_template**2))
    
    if template_norm == 0:
        return np.zeros_like(data_cube), noise_spectrum

    # 3. Process each pixel
    snr_cube = np.zeros_like(data_cube)
    for i in tqdm(range(data_cube.shape[0]), desc="Processing Pixels"):
        spectrum = data_cube[i, :]
        whitened_spectrum = spectrum / noise_spectrum
        
        # Perform the whitened correlation
        correlation = np.correlate(whitened_spectrum, whitened_template, mode='same')
        
        # Normalize to get the unaligned SNR spectrum
        snr_unaligned = correlation / template_norm
        
        # --- 4. Explicit Alignment Logic ---
        # Find the index of the most prominent peak (positive or negative)
        peak_idx_corr = np.argmax(np.abs(snr_unaligned))
        peak_idx_temp = np.argmax(np.abs(template))
        
        # Calculate the shift required to move the correlation peak
        shift_amount = peak_idx_corr - peak_idx_temp
        
        # Roll the array to perform the alignment
        snr_aligned = np.roll(snr_unaligned, -shift_amount)
        
        snr_cube[i, :] = snr_aligned

    if enforce_positivity:
        print("  - Enforcing positivity on final SNR cube.")
        snr_cube[snr_cube < 0] = 0
        
    return snr_cube, noise_spectrum
    
    
def run_whitened_correlation_search(
    data_cube: np.ndarray,
    template_bank: list[dict],
    noise_spectrum: np.ndarray = None,
    enforce_positivity: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a whitened matched filter search using an entire template bank
    and linearly averages the results for accurate localization.

    This is the recommended, best-practice function for a full search.

    Args:
        data_cube: The 2D data cube (pixels x frequencies) to be searched.
        template_bank: A list of pre-filtered template dictionaries.
        noise_spectrum: A pre-calculated 1D noise spectrum. If None, it will
                        be estimated using a robust MAD estimator.
        enforce_positivity: If True, sets negative SNR values to zero to aid
                            in candidate finding.

    Returns:
        A tuple containing the final SNR cube and the noise spectrum.
    """
    print("--- Running Full Template Bank Whitened Correlation Search ---")

    # 1. Estimate noise if not provided
    if noise_spectrum is None:
        print("  - Estimating noise with robust MAD estimator...")
        median_abs_dev = np.nanmedian(np.abs(data_cube - np.nanmedian(data_cube, axis=0)), axis=0)
        noise_spectrum = median_abs_dev * 1.4826
    
    noise_floor = 1e-9
    noise_spectrum[np.isnan(noise_spectrum) | (noise_spectrum <= noise_floor)] = noise_floor

    # 2. Process each pixel
    snr_cube = np.zeros_like(data_cube)
    n_channels = data_cube.shape[1]
    
    for i in tqdm(range(data_cube.shape[0]), desc="Processing Pixels"):
        spectrum = data_cube[i, :]
        
        # Accumulators for linearly averaging results
        snr_sum_spectrum = np.zeros(n_channels)
        snr_count_spectrum = np.zeros(n_channels, dtype=int)

        # Pre-whiten the data for this pixel
        whitened_spectrum = spectrum / noise_spectrum

        # 3. Iterate through each template in the bank
        for temp_info in template_bank:
            full_template = np.zeros(n_channels)
            start, end = temp_info['start'], temp_info['end']
            full_template[start:end] = temp_info['prof']

            whitened_template = full_template / noise_spectrum
            template_norm = np.sqrt(np.sum(whitened_template**2))
            if template_norm == 0:
                continue

            # Perform correlation and normalize
            correlation = np.correlate(whitened_spectrum, whitened_template, mode='same')
            snr_spectrum_result = correlation / template_norm
            
            # Add the result to the accumulators
            snr_sum_spectrum += snr_spectrum_result
            snr_count_spectrum += 1

        # 4. Calculate the final averaged SNR spectrum for the pixel
        if np.any(snr_count_spectrum > 0):
            valid_mask = snr_count_spectrum > 0
            snr_cube[i, valid_mask] = snr_sum_spectrum[valid_mask] / snr_count_spectrum[valid_mask]

    # 5. Enforce positivity on the final cube
    if enforce_positivity:
        snr_cube[snr_cube < 0] = 0
        
    return snr_cube, noise_spectrum


def refine_candidate_localization(
    candidate_pixel_idx: int,
    data_cube: np.ndarray,
    template_bank: list[dict],
    noise_spectrum: np.ndarray
) -> tuple[int, int, float]:
    """
    Refines the frequency localization of a coarse candidate by finding the
    best-matching template and running a full whitened correlation.

    Args:
        candidate_pixel_idx: The pixel index of the candidate to refine.
        data_cube: The 2D (filtered) data cube.
        template_bank: The full list of template dictionaries.
        noise_spectrum: The 1D per-channel noise spectrum.

    Returns:
        A tuple containing the refined location and SNR:
        (pixel_idx, refined_freq_idx, refined_snr)
    """
    spectrum = data_cube[candidate_pixel_idx, :]

    # --- 1. Find the best-matching template for this spectrum ---
    best_snr = -np.inf
    best_template_info = None

    for temp_info in template_bank:
        start, end = temp_info['start'], temp_info['end']
        template_profile = temp_info['prof']
        
        # Perform a zero-lag whitened check to find the best template
        data_segment = spectrum[start:end]
        noise_segment = noise_spectrum[start:end]
        
        norm_effective = np.sqrt(np.sum((template_profile / noise_segment)**2))
        if norm_effective < 1e-9:
            continue
            
        weighted_data = data_segment / noise_segment**2
        snr = np.sum(weighted_data * template_profile) / norm_effective

        if snr > best_snr:
            best_snr = snr
            best_template_info = temp_info

    if best_template_info is None:
        return candidate_pixel_idx, -1, -1 # Could not find a matching template

    # --- 2. Run a full whitened correlation with the best template ---
    # Create the full-length version of the best template found
    n_channels = len(spectrum)
    best_template_full = np.zeros(n_channels)
    start, end = best_template_info['start'], best_template_info['end']
    best_template_full[start:end] = best_template_info['prof']
    
    # Use the robust run_whitened_snr_spectrum function to get the localized spectrum
    localized_snr_spectrum = run_whitened_snr_spectrum(
        spectrum=spectrum,
        template=best_template_full,
        noise_spectrum=noise_spectrum
    )

    # --- 3. Find the peak of the localized SNR spectrum ---
    refined_freq_idx = np.argmax(localized_snr_spectrum)
    refined_snr = localized_snr_spectrum[refined_freq_idx]

    return candidate_pixel_idx, refined_freq_idx, refined_snr

    
def run_matched_filter_search(
    data_cube: np.ndarray,
    templates: Union[Dict, List[Dict]],
    noise_spectrum: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs a matched filter search over a data cube using a template or bank.

    This function iterates through each spectrum (pixel) in the data cube and
    applies a whitened matched filter using the provided template(s). It is
    optimized for templates that are shorter than the full spectrum and
    assumes a zero-lag alignment (direct match).

    Parameters
    ----------
    data_cube : np.ndarray
        The 2D data cube (pixels x frequencies) to be searched. This data
        should already be foreground-filtered.
    templates : Union[Dict, List[Dict]]
        The template(s) to search for. Can be a single template dictionary
        or a list of dictionaries (a template bank). Each dictionary must
        contain 'prof', 'start', and 'end' keys.
    noise_spectrum : np.ndarray, optional
        A pre-calculated 1D array of the per-channel noise standard deviation.
        If not provided, it will be estimated from the `data_cube` using
        `np.nanstd`. Defaults to None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - snr_cube (np.ndarray): A 2D array with the same shape as `data_cube`,
          containing the peak SNR found at each channel for each pixel.
        - noise_spectrum (np.ndarray): The 1D per-channel noise standard
          deviation used for the calculation.
    """
    print("--- Running Matched Filter Search ---")

    # --- 1. Estimate the noise PER CHANNEL if not provided ---
    if noise_spectrum is None:
        print("Estimating noise spectrum from the data cube...")
        noise_spectrum = np.nanstd(data_cube, axis=0)
        # Handle potential NaNs or zeros in the noise estimate
        noise_spectrum[np.isnan(noise_spectrum) | (noise_spectrum <= 0)] = 1e-9

    # --- 2. Normalize input to handle both single template and bank ---
    if isinstance(templates, dict):
        template_bank = [templates] # Wrap a single template in a list
    else:
        template_bank = templates

    # --- 3. Run the fast, direct-match search ---
    snr_cube = np.zeros_like(data_cube)
    n_pixels = data_cube.shape[0]

    for i in tqdm(range(n_pixels), desc="Processing Pixels"):
        spectrum = data_cube[i, :]
        pixel_snr_spectrum = np.zeros_like(spectrum)

        # Slide each template from the bank across the spectrum
        for temp_info in template_bank:
            start = temp_info['start']
            end = temp_info['end']
            template_profile = temp_info['prof']

            # Get the slice of the data and noise corresponding to this template
            data_segment = spectrum[start:end]
            noise_segment = noise_spectrum[start:end]

            # Calculate the effective normalization factor for the whitened template
            norm_effective = np.sqrt(np.sum((template_profile / noise_segment)**2))
            if norm_effective < 1e-6:
                continue

            # Correlate the whitened data with the template
            weighted_data = data_segment / noise_segment**2
            snr = np.sum(weighted_data * template_profile) / norm_effective

            # "Paint" the single SNR value across the template's footprint if it's an improvement
            current_snr_segment = pixel_snr_spectrum[start:end]
            update_mask = snr > current_snr_segment
            pixel_snr_spectrum[start:end][update_mask] = snr

        snr_cube[i, :] = pixel_snr_spectrum

    print("\nSNR cube generation complete.")
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


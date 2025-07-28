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
    # normalize the template
    template_norm = template/template.max()

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

def run_fft_filter_cube(
    data_cube: np.ndarray,
    template: np.ndarray,
    align_output: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a simple matched filter on a data cube using FFT correlation.

    This function processes a cube pixel-by-pixel. It uses the template as-is
    with no modifications (e.g., no mean-subtraction). It calculates the
    per-channel noise spectrum but uses a single noise estimate per pixel
    for the FFT normalization.

    Args:
        data_cube: The 2D (pixels x frequencies) data cube to filter.
        template: The 1D signal template to search for.
        align_output: If True, attempts to align the output SNR spectrum.

    Returns:
        A tuple containing:
        - snr_cube (np.ndarray): The 2D cube of filtered SNR spectra.
        - noise_spectrum (np.ndarray): The calculated per-channel noise.
    """
    print("--- Running Simple FFT Filter on Cube ---")
    
    # --- 1. Calculate the per-channel noise spectrum from the cube ---
    noise_spectrum = np.nanstd(data_cube, axis=0)
    # Handle potential NaNs or zeros in the noise estimate
    noise_spectrum[np.isnan(noise_spectrum) | (noise_spectrum <= 0)] = 1e-9

    # Prepare output array
    snr_cube = np.zeros_like(data_cube)
    n_pixels = data_cube.shape[0]

    # --- 2. Process the Template (as-is) ---
    # Use the template directly without modification as requested
    template_energy = np.sum(template**2)
    if template_energy == 0:
        return snr_cube, noise_spectrum

    # --- 3. Loop through each pixel in the cube ---
    for i in tqdm(range(n_pixels), desc="Filtering Pixels"):
        spectrum = data_cube[i, :]

        # For FFT normalization, a single noise sigma per spectrum is used.
        # noise_sigma_pixel = 
        # if noise_sigma_pixel <= 0:
        #     continue

        # Correlate the spectrum with the raw template
        correlation = np.correlate(spectrum, template, mode='same')
        snr_unaligned = (correlation / (np.sqrt(template_energy) * noise_spectrum))

        # Align the output as before
        if align_output:
            correlation_peak_index = np.argmax(np.abs(snr_unaligned))
            template_peak_index = np.argmax(np.abs(template))
            shift_amount = correlation_peak_index - template_peak_index
            snr_spectrum = np.roll(snr_unaligned, -shift_amount)
        else:
            snr_spectrum = snr_unaligned

        snr_cube[i, :] = np.abs(snr_spectrum)

    return snr_cube, noise_spectrum


def run_whitened_snr_spectrum(
    spectrum: np.ndarray,
    template: np.ndarray,
    noise_spectrum: np.ndarray
) -> np.ndarray:
    """
    Performs a whitened matched filter and returns the full SNR spectrum.

    This function whitens the data and template by the per-channel noise
    before correlating them, providing a statistically robust SNR spectrum.

    Args:
        spectrum: The 1D array of the observed (filtered) spectrum.
        template: The 1D array of the (filtered) signal template.
        noise_spectrum: The 1D array of the per-channel noise standard deviation.

    Returns:
        A 1D array representing the full whitened SNR spectrum.
    """
    # Ensure noise has a floor to prevent division by zero
    noise_floor = 1e-9
    noise_spectrum[noise_spectrum <= noise_floor] = noise_floor

    # --- 1. Whiten the data and the template ---
    whitened_spectrum = spectrum / noise_spectrum
    whitened_template = template / noise_spectrum

    # --- 2. Calculate the normalization factor ---
    # This is the standard deviation of the whitened template
    template_norm = np.sqrt(np.sum(whitened_template**2))
    if template_norm == 0:
        return np.zeros_like(spectrum)

    # --- 3. Correlate the whitened arrays ---
    # This is the whitened matched filter operation
    correlation = np.correlate(whitened_spectrum, whitened_template, mode='same')

    # --- 4. Normalize to get the final SNR spectrum ---
    snr_spectrum = correlation / template_norm
    
    return snr_spectrum
    

def run_linear_fft_bank_search(
    data_cube: np.ndarray,
    template_bank: List[Dict],
    noise_spectrum: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs a linear matched filter search using an FFT-based correlation.

    Instead of taking the maximum SNR, this function averages the output of
    all templates that cover a given frequency channel. This is a linear
    operation that preserves the full filter response, including negative values.

    Parameters
    ----------
    data_cube : np.ndarray
        The 2D data cube (pixels x frequencies) to be searched.
    template_bank : List[Dict]
        A list of pre-filtered template dictionaries.
    noise_spectrum : np.ndarray, optional
        A 1D array of the per-channel noise standard deviation. If not
        provided, it will be estimated from the data_cube. Defaults to None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - snr_cube (np.ndarray): The 2D cube of linearly filtered spectra.
        - noise_spectrum (np.ndarray): The 1D noise spectrum used.
    """
    print("--- Running Linear FFT-based Bank Search (Averaging) ---")

    # --- 1. Estimate noise if not provided ---
    if noise_spectrum is None:
        print("  - Estimating noise spectrum from the data cube...")
        noise_spectrum = np.nanstd(data_cube, axis=0)
        noise_spectrum[np.isnan(noise_spectrum) | (noise_spectrum <= 0)] = 1e-9

    snr_cube = np.zeros_like(data_cube)
    n_pixels, n_channels = data_cube.shape

    for i in tqdm(range(n_pixels), desc="Processing Pixels (Linear FFT)"):
        spectrum = data_cube[i, :]
        
        # Accumulators for summing results and counting contributions
        snr_sum_spectrum = np.zeros(n_channels, dtype=float)
        snr_count_spectrum = np.zeros(n_channels, dtype=int)

        for temp_info in template_bank:
            start, end = temp_info['start'], temp_info['end']
            template_profile = temp_info['prof']

            # Create a full-sized template for correlation
            full_template = np.zeros(n_channels)
            full_template[start:end] = template_profile

            template_energy = np.sum(full_template**2)
            local_noise_sigma = np.mean(noise_spectrum[start:end])

            if template_energy == 0 or local_noise_sigma <= 0:
                continue

            # Correlate and get the full SNR spectrum for this template
            correlation = np.correlate(spectrum, full_template, mode='same')
            snr_spectrum_full = correlation / (np.sqrt(template_energy) * local_noise_sigma)

            # --- Linear Combination Step ---
            # Add the result to the sum accumulator
            snr_sum_spectrum[start:end] += snr_spectrum_full[start:end]
            # Increment the count for the channels that were covered
            snr_count_spectrum[start:end] += 1

        # Calculate the final averaged SNR spectrum for the pixel
        # Avoid division by zero for channels that were never covered
        valid_counts = snr_count_spectrum > 0
        snr_cube[i, valid_counts] = snr_sum_spectrum[valid_counts] / snr_count_spectrum[valid_counts]

    print("\nLinearly filtered cube generation complete.")
    return snr_cube, noise_spectrum
    

# this is a whitened matched filter (might be better?)
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


def run_subspace_matched_filter(
    data_cube: np.ndarray,
    templates: Union[Dict, List[Dict]],
    noise_covariance: np.ndarray,
    num_modes_to_subtract: int = 5,
    edge_trim_channels: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs an intermediate matched filter using subspace projection (PCA),
    operating on the central part of the band.
    """
    print(f"--- Running Subspace Matched Filter (Subtracting {num_modes_to_subtract} Modes) ---")

    # --- Step 1: Find the dominant noise modes ---
    print("Finding dominant noise modes from covariance matrix...")
    eigenvalues, eigenvectors = np.linalg.eigh(noise_covariance)
    top_modes = eigenvectors[:, -num_modes_to_subtract:]

    # --- Step 2: Clean the data cube by subtracting the noise modes ---
    # We will only modify the central, trimmed part of the data.
    data_cube_cleaned = np.copy(data_cube)
    channel_slice = slice(edge_trim_channels, -edge_trim_channels)

    for i in tqdm(range(data_cube.shape[0]), desc="Cleaning Data with PCA"):
        # Extract the full spectrum
        spectrum_full = data_cube[i, :]
        # Trim it to match the dimensions of the noise modes
        spectrum_trimmed = spectrum_full[channel_slice]
        
        # Now the dot product will work correctly
        coeffs = np.dot(spectrum_trimmed, top_modes)
        noise_model = np.dot(top_modes, coeffs)
        
        # Subtract the noise model to get the cleaned, trimmed spectrum
        cleaned_spectrum_trimmed = spectrum_trimmed - noise_model
        
        # Place the cleaned segment back into the full-sized cube
        data_cube_cleaned[i, channel_slice] = cleaned_spectrum_trimmed

    # --- Step 3: Run the standard whitened matched filter on the cleaned data ---
    snr_cube, final_noise_spectrum = run_matched_filter_search(
        data_cube=data_cube_cleaned,
        templates=templates
    )
    
    return snr_cube, final_noise_spectrum

    
def run_generalized_matched_filter(
    data_cube: np.ndarray,
    templates: Union[Dict, List[Dict]],
    noise_covariance: np.ndarray,
    regularization: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs a Generalized Matched Filter search using a regularized noise
    covariance matrix for numerical stability.

    Parameters
    ----------
    data_cube : np.ndarray
        The 2D data cube (pixels x frequencies) to be searched.
    templates : Union[Dict, List[Dict]]
        The template(s) to search for.
    noise_covariance : np.ndarray
        The 2D (N_CHANNELS x N_CHANNELS) noise covariance matrix, C_n.
    regularization : float, optional
        A small value added to the diagonal of the covariance matrix to
        ensure it is well-conditioned before inversion. Defaults to 1e-5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - snr_cube (np.ndarray): The resulting 2D SNR cube.
        - C_n_inv (np.ndarray): The inverse of the regularized covariance matrix.
    """
    print("--- Running Generalized Matched Filter Search ---")

    # --- 1. Regularize and Invert the Covariance Matrix ---
    print(f"Regularizing and inverting the noise covariance matrix...")
    
    # Add a small epsilon to the diagonal for numerical stability
    C_n_reg = noise_covariance + regularization * np.eye(noise_covariance.shape[0])
    
    # Now invert the stabilized matrix
    C_n_inv = np.linalg.pinv(C_n_reg)

    # (The rest of the function is identical to before)
    if isinstance(templates, dict):
        template_bank = [templates]
    else:
        template_bank = templates
        
    snr_cube = np.zeros_like(data_cube)
    n_pixels = data_cube.shape[0]

    for i in tqdm(range(n_pixels), desc="Processing Pixels (Generalized)"):
        spectrum = data_cube[i, :]
        pixel_snr_spectrum = np.zeros_like(spectrum)

        for temp_info in template_bank:
            start, end = temp_info['start'], temp_info['end']
            C_n_inv_segment = C_n_inv[start:end, start:end]
            s = temp_info['prof']
            d = spectrum[start:end]
            
            norm_sq = s.T @ C_n_inv_segment @ s
            if norm_sq < 1e-6: continue
            norm = np.sqrt(norm_sq)
            
            score = s.T @ C_n_inv_segment @ d
            snr = score / norm
            
            current_snr_segment = pixel_snr_spectrum[start:end]
            update_mask = snr > current_snr_segment
            pixel_snr_spectrum[start:end][update_mask] = snr
            
        snr_cube[i, :] = pixel_snr_spectrum

    print("\nSNR cube generation complete.")
    return snr_cube, C_n_inv

    
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


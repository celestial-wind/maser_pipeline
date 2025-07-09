# ohm_search_simulator.py (Regenerated Version)

"""
The core orchestration module for the OHM search simulation pipeline.

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

# External library dependencies
import healpy as hp
from pygdsm import GlobalSkyModel
import uvtools

# Local module dependencies
import ohm_template_generator as otg

# =============================================================================
# --- Component Functions: RFI and Foreground Filtering ---
# =============================================================================

def generate_realistic_rfi_mask(freqs: np.ndarray) -> np.ndarray:
    """
    Generates a stationary RFI mask based on known interfering frequency bands.

    Args:
        freqs: The array of channel center frequencies in MHz.

    Returns:
        A 1D weight array (0 for flagged channels, 1 for clean channels).
    """
    print("  - Generating realistic stationary RFI mask...")
    weights = np.ones_like(freqs)
    
    # Approximate bands for North America. Can be refined with real data.
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
    num_random_flags = int(0.01 * len(freqs))
    random_indices = np.random.choice(np.where(weights == 1)[0], num_random_flags, replace=False)
    weights[random_indices] = 0
    
    return weights

# Add this function to the filtering section
# def apply_simple_delay_filter(
#     spectrum: np.ndarray,
#     weights: np.ndarray,
#     delay_notch_width: int = 15
# ) -> np.ndarray:
#     """
#     Simulates a basic delay filter by notching out the first few Fourier modes.
#     This is a robust, NumPy-only alternative to using uvtools.

#     Args:
#         spectrum: The 1D input data slice.
#         weights: The weights array (0 for flagged channels).
#         delay_notch_width: The number of Fourier modes to zero out on each side
#                            of the delay spectrum. Larger values are more aggressive.

#     Returns:
#         The real part of the filtered spectrum.
#     """
#     # Go to delay space (Fourier space)
#     delay_spectrum = np.fft.fft(spectrum * weights)
    
#     # Notch out the low-delay (spectrally smooth) modes
#     delay_spectrum[:delay_notch_width] = 0
#     delay_spectrum[-delay_notch_width:] = 0
    
#     # Return to frequency space
#     filtered_spectrum = np.fft.ifft(delay_spectrum)
    
#     return filtered_spectrum.real

# # In ohm_search_simulator.py

def apply_simple_delay_filter(
    spectrum: np.ndarray,
    weights: np.ndarray,
    delay_notch_width: int = 15
) -> np.ndarray:
    """
    Simulates a basic delay filter by notching out the first few Fourier modes.
    This is a robust, NumPy-only alternative to using uvtools.
    """
    delay_spectrum = np.fft.fft(spectrum * weights)
    
    delay_spectrum[:delay_notch_width] = 0
    delay_spectrum[-delay_notch_width:] = 0
    
    filtered_spectrum = np.fft.ifft(delay_spectrum)
    
    return filtered_spectrum.real


def apply_physical_delay_filter(
    spectrum: np.ndarray,
    weights: np.ndarray,
    freqs_mhz: np.ndarray,
    delay_cut_ns: float
) -> np.ndarray:
    """
    Applies a delay filter using a physical cut in nanoseconds. This function
    calculates the correct notch width and calls the simple filter.
    """
    # Calculate the total bandwidth in Hz
    bandwidth_hz = (np.max(freqs_mhz) - np.min(freqs_mhz)) * 1e6
    if bandwidth_hz == 0:
        return spectrum
        
    # Convert the desired delay cut into an integer number of channels
    delay_cut_s = delay_cut_ns * 1e-9
    notch_width = int(np.round(delay_cut_s * bandwidth_hz))
    
    # Apply the simple filter with the calculated notch width
    return apply_simple_delay_filter(spectrum, weights, delay_notch_width=notch_width)

# In ohm_search_simulator.py, replace your old filter functions with this one

from scipy.signal.windows import tukey

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
    delay_spectrum = np.fft.fft(spectrum * weights)
    filtered_delay_spectrum = delay_spectrum * fft_filter
    filtered_spectrum = np.fft.ifft(filtered_delay_spectrum)
    
    return filtered_spectrum.real
    
def apply_uvtools_dayenu_filter(
    spectrum: np.ndarray, 
    weights: np.ndarray, 
    filter_scale_kpar: float = 0.05
) -> np.ndarray:
    """
    Applies a Dayenu delay filter using the uvtools library for high fidelity.

    Args:
        spectrum: The 1D input data slice.
        weights: The weights array (0 for flagged channels).
        filter_scale_kpar: Filtering scale in units of k_parallel. Larger values
                           are more aggressive.

    Returns:
        The real part of the filtered spectrum.
    """
    # Ensure data is complex type for FFT operations
    if not np.iscomplexobj(spectrum):
        spectrum = spectrum.astype(np.complex128)

    # Convert to delay space
    delay_spectrum = np.fft.fft(spectrum * weights)
    
    # Apply the filter. It returns the cleaned data and an info dictionary.
    # We only need the cleaned data for this simulation.
    cleaned_delay_spectrum, info = uvtools.dspec.dayenu_filter(
        delay_spectrum, 
        weights, 
        filter_scale_kpar=filter_scale_kpar
    )
    
    # Inverse transform the cleaned delay spectrum to get the filtered spectrum
    filtered_spectrum = np.fft.ifft(cleaned_delay_spectrum)
    
    return filtered_spectrum.real

# =============================================================================
# --- Component Functions: Sky and Weight Generation ---
# =============================================================================

def generate_realistic_sky_flux_map(
    num_pixels: int, confusion_sigma: float = 0.1, source_flux_min: float = 0.3, 
    gamma: float = 2.5, n_faint_sources: int = 2000
) -> np.ndarray:
    """Generates a realistic map of faint sky signals (confusion + sources)."""
    base_map = np.random.normal(0, confusion_sigma, num_pixels)
    u = np.random.uniform(0, 1, n_faint_sources)
    source_fluxes = source_flux_min * (1 - u)**(-1 / (gamma - 1))
    source_pixels = np.random.choice(num_pixels, n_faint_sources, replace=True)
    np.add.at(base_map, source_pixels, source_fluxes)
    return base_map

def generate_sky_weights(grid_shape: Tuple[int, int]) -> np.ndarray:
    """Generates a realistic 2D map of sky weights for sensitivity variations."""
    ny, nx = grid_shape
    x, y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    beam = np.exp(-(x**2 + y**2) / (2 * 0.6**2))
    galaxy_plane = 1.0 - 0.7 * np.exp(-(y - 0.1)**2 / (2 * 0.15**2))
    
    total_weights = beam * galaxy_plane
    total_weights[total_weights < 0] = 0
    total_weights /= np.max(total_weights)
    min_weight = 1e-4
    total_weights[total_weights < min_weight] = min_weight
    return total_weights
#
# NEW function to be added to ohm_search_simulator.py
#
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
# =============================================================================
# --- Simulation Orchestration ---
# =============================================================================

def generate_sky_model_slice(model_type: str, num_pixels: int, freqs: np.ndarray) -> np.ndarray:
    """Dispatcher to generate a 1D slice of a sky model at a reference frequency."""
    print(f"  - Generating '{model_type}' sky model.")
    if model_type == 'blank':
        return np.zeros(num_pixels)
    elif model_type == 'powerlaw_sources':
        return generate_realistic_sky_flux_map(num_pixels)
    elif model_type == 'gdsm':
        grid_size = int(np.sqrt(num_pixels))
        if grid_size**2 != num_pixels: raise ValueError("GDSM model needs a perfect square num_pixels.")
        ref_freq = freqs[len(freqs) // 2]
        gsm = GlobalSkyModel(freq_unit='MHz')
        healpix_map = gsm.generate(ref_freq)
        projected_map = hp.gnomview(
            healpix_map, rot=(120, 40, 0), xsize=grid_size, ysize=grid_size,
            reso=1.5 * 60 / grid_size, return_projected_map=True, no_plot=True
        )
        return projected_map.flatten()
    else:
        raise ValueError(f"Unknown sky_model: {model_type}")

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

# def generate_sky_image_cube(
#     num_pixels: int, 
#     freqs: np.ndarray, 
#     noise_sigma_base: float = 0.5,
#     num_injections: int = 100, 
#     sky_model: str = 'powerlaw_sources',
#     noise_model: str = 'beam_weighted',
#     add_foregrounds: bool = True
# ) -> Tuple[np.ndarray, Dict, np.ndarray]:
#     """
#     Generates a simulated sky data cube with selectable physical models.

#     Args:
#         sky_model: 'blank', 'powerlaw_sources', or 'gdsm'.
#         noise_model: 'uniform' or 'beam_weighted'.
#         Other args are described in previous versions.

#     Returns:
#         A tuple of (data_cube, ground_truth_dict, sky_weights_map).
#     """
#     num_channels = len(freqs)
#     print(f"Generating simulation cube with {num_pixels} pixels...")

#     # Generate base sky and instrumental weights based on selected models
#     base_sky_slice = generate_sky_model_slice(sky_model, num_pixels, freqs)
#     sky_weights = generate_instrument_weights(noise_model, num_pixels)
    
#     # Tile the 1D sky slice into a 2D (pixel, frequency) cube
#     data_cube = np.tile(base_sky_slice[:, np.newaxis], (1, num_channels))

#     # Add spectrally-smooth foregrounds (optional)
#     if add_foregrounds:
#         print("  - Adding power-law foregrounds...")
#         nu_ref, alpha = freqs[len(freqs) // 2], -2.5
#         foreground_amplitude = np.random.uniform(20, 50) * noise_sigma_base
#         foreground_shape = (freqs / nu_ref)**alpha
#         data_cube += foreground_amplitude * foreground_shape[np.newaxis, :]

#     # Inject synthetic OHM signals
#     ground_truth = {'injections': []}
#     vel_axis = np.linspace(-1200, 1200, 4096)
#     injection_indices = np.random.choice(num_pixels, num_injections, replace=False)

#     for pixel_idx in tqdm(injection_indices, desc="Injecting Signals"):
#         # the amplitude of the injections could be made more like a luminosity function
#         # to approximate this we can be it the absolute value of a gaussian distribution  
#         z_inject, amp_inject = np.random.uniform(1.2, 3.0), np.abs(np.random.normal(loc=2, scale=3)) #np.random.uniform(0.1, 10.0)
#         template, start_idx, end_idx = otg.generate_final_template(z_inject, vel_axis, freqs)
#         if template.size == 0 or np.max(template) == 0: continue

#         # added noiseless profile
#         noiseless_profile = np.zeros_like(freqs)
#         scaled_template = template * (amp_inject / np.max(template))
#         noiseless_profile[start_idx:end_idx] = scaled_template
        
#         g_truth_entry = {'pixel_index': pixel_idx, 'z': z_inject, 'amp': amp_inject, 'noiseless_profile': noiseless_profile}
#         ground_truth['injections'].append(g_truth_entry)
#         data_cube[pixel_idx, start_idx:end_idx] += template * (amp_inject / np.max(template))

#     # Add instrumental noise, scaled by the weights
#     print("  - Adding instrumental noise...")
#     print(f" Noise sigma is: [ {noise_sigma_base} ]")
#     for i in tqdm(range(num_pixels), desc="Adding Noise"):
#         # Noise is higher where weights (sensitivity) are lower
#         pixel_noise_sigma = noise_sigma_base / sky_weights[i]
#         noise = np.random.normal(0, pixel_noise_sigma, num_channels)
#         data_cube[i, :] += noise
        
#     return data_cube, ground_truth, sky_weights
#
# UPDATED generate_sky_image_cube function in ohm_search_simulator.py
#
def generate_sky_image_cube(
    num_pixels: int, 
    freqs: np.ndarray, 
    noise_sigma_base: float = 0.5,
    num_injections: int = 100, 
    sky_model: str = 'powerlaw_sources',
    noise_model: str = 'uniform',
    # Note: add_foregrounds is no longer needed, as GDSM includes them
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
    intrinsic_template_v = otg.create_intrinsic_template(vel_axis_kms=vel_axis, N_population=5000, verbose=False)

    for pixel_idx in tqdm(injection_indices, desc="Injecting Signals"):
        z_inject = np.random.uniform(1.1, 3.1)
        amp_inject = np.abs(np.random.normal(loc=7, scale=2))
        
        # Transform the master template for this specific injection
        template, start_idx, end_idx = otg.process_to_native_resolution(
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

def generate_delay_matched_boxcar(freqs_mhz: np.ndarray, delay_cut_ns: float) -> np.ndarray:
    """
    Generates an advanced, physically-motivated boxcar template whose width
    is scaled by the delay cut of the foreground filter.

    The shape is a positive central box flanked by two negative boxes, which
    mimics the shape of a signal after being high-pass filtered.

    Args:
        freqs_mhz: The frequency axis in MHz.
        delay_cut_ns: The delay cut of the filter, in nanoseconds.

    Returns:
        The generated 1D template array.
    """
    # The characteristic width of a feature shaped by a delay filter is ~1 / tau_cut
    delay_cut_s = delay_cut_ns * 1e-9
    feature_width_hz = 1.0 / delay_cut_s
    
    # Convert this width in Hz to a width in number of channels
    channel_width_hz = (freqs_mhz[1] - freqs_mhz[0]) * 1e6
    feature_width_chans = int(np.round(feature_width_hz / channel_width_hz))
    
    # Ensure the width is an odd number for symmetry
    if feature_width_chans % 2 == 0:
        feature_width_chans += 1
        
    print(f"Delay cut of {delay_cut_ns} ns -> Matched template width of {feature_width_chans} channels.")

    # Create the template with a central positive part and negative sidelobes
    # The sidelobes will be half the width of the central part
    center_width = feature_width_chans
    sidelobe_width = center_width // 2
    
    # The amplitude of the negative parts is set to make the total sum of the template zero
    positive_part = np.ones(center_width)
    negative_amplitude = - (center_width / (2 * sidelobe_width))
    negative_part = negative_amplitude * np.ones(sidelobe_width)
    
    template = np.concatenate([negative_part, positive_part, negative_part])
    
    return template

    
# =============================================================================
# --- Search and Filtering Algorithms ---
# =============================================================================

def run_matched_filter(spectrum: np.ndarray, template: np.ndarray, weights: np.ndarray, noise_sigma: float) -> float:
    """Calculates the SNR of a signal at a specific, known alignment."""
    template_norm = template - np.mean(template)
    template_energy = np.sum(template_norm**2 * weights)
    if template_energy == 0 or noise_sigma <= 0: return 0.0
    score = np.sum(spectrum * template_norm * weights)
    return score / (np.sqrt(template_energy) * noise_sigma)

def run_blind_search(spectrum: np.ndarray, template: np.ndarray, noise_sigma: float) -> np.ndarray:
    """Performs a blind search by sliding a template across a spectrum."""
    template_norm = template - np.mean(template)
    template_energy = np.sum(template_norm**2)
    if template_energy == 0 or noise_sigma <= 0: return np.zeros_like(spectrum)
    correlation = correlate(spectrum, template_norm, mode='same')
    snr_spectrum = correlation / (np.sqrt(template_energy) * noise_sigma)
    return snr_spectrum

def run_threshold_search(filtered_spectrum: np.ndarray, noise_per_channel: np.ndarray) -> float:
    """Finds the peak significance of any single channel. A baseline search method."""
    significance_spectrum = np.zeros_like(filtered_spectrum)
    valid_indices = noise_per_channel > 0
    significance_spectrum[valid_indices] = filtered_spectrum[valid_indices] / noise_per_channel[valid_indices]
    return np.max(significance_spectrum)

def run_pipeline_aware_matched_filter(
    spectrum: np.ndarray, freqs: np.ndarray, intrinsic_template_v: np.ndarray,
    vel_axis_kms: np.ndarray, rfi_weights: np.ndarray, noise_per_channel: np.ndarray,
    z_search_list: np.ndarray, dayenu_filter_scale: float = 0.05
) -> Tuple[float, float, np.ndarray]:
    """
    Performs an optimized matched-filter search using "pipeline-aware" templates.
    (See previous response for detailed explanation).
    """
    best_snr, best_z, best_template = -np.inf, None, None
    for z_trial in z_search_list:
        ideal_template, start, end = otg.process_to_native_resolution(
            intrinsic_template_v, vel_axis_kms, z_trial, freqs
        )
        if ideal_template.size == 0: continue
        
        template_full = np.zeros_like(freqs)
        template_full[start:end] = ideal_template
        template_distorted = apply_simple_delay_filter(
            template_full * rfi_weights,
            rfi_weights,
            delay_notch_width=15 # This is now the control parameter
        )
        
        template_slice = template_distorted[start:end]
        if np.sum(template_slice**2) == 0: continue
            
        weights_slice = rfi_weights[start:end]
        eff_noise = np.mean(noise_per_channel[start:end][weights_slice == 1])
        snr = run_matched_filter(spectrum[start:end], template_slice, weights_slice, eff_noise)

        if snr > best_snr:
            best_snr, best_z, best_template = snr, z_trial, template_slice
            
    return best_snr, best_z, best_template

def run_fine_fit_search(
    spectrum: np.ndarray, freqs: np.ndarray, intrinsic_template_v: np.ndarray,
    vel_axis_kms: np.ndarray, rfi_weights: np.ndarray, noise_per_channel: np.ndarray,
    z_search_list: np.ndarray
) -> Tuple[float, float, np.ndarray, int, int]:
    """
    A dedicated search function for parameter fitting.

    This is a copy of the pipeline-aware search but is designed specifically
    for fitting and returns the start and end indices of the best-fit template,
    in addition to the other parameters. This leaves the original search
    function untouched for the ROC analysis.

    Returns:
        A tuple of (best_snr, best_z, best_template, best_start_idx, best_end_idx).
    """
    best_snr, best_z, best_template = -np.inf, None, None
    best_start_idx, best_end_idx = -1, -1

    for z_trial in z_search_list:
        ideal_template, start, end = otg.process_to_native_resolution(
            intrinsic_template_v, vel_axis_kms, z_trial, freqs
        )
        if ideal_template.size == 0: continue
        
        template_full = np.zeros_like(freqs)
        template_full[start:end] = ideal_template
        template_distorted = apply_simple_delay_filter(
            template_full * rfi_weights, rfi_weights
        )
        
        template_slice = template_distorted[start:end]
        if np.sum(template_slice**2) == 0: continue
            
        weights_slice = rfi_weights[start:end]
        eff_noise = np.mean(noise_per_channel[start:end][weights_slice == 1])
        snr = run_matched_filter(spectrum[start:end], template_slice, weights_slice, eff_noise)

        if snr > best_snr:
            best_snr, best_z, best_template = snr, z_trial, template_slice
            best_start_idx, best_end_idx = start, end

    return best_snr, best_z, best_template, best_start_idx, best_end_idx
    
# --- High-Level Search Strategies ---

def z_to_freq(z, rest_freq=1667.359): return rest_freq / (1 + z)
def freq_to_z(freq, rest_freq=1667.359): return (rest_freq / freq) - 1

def run_redshift_straddle_search(
    spectrum: np.ndarray, freqs: np.ndarray, intrinsic_template_v: np.ndarray,
    vel_axis_kms: np.ndarray, rfi_weights: np.ndarray, noise_per_channel: np.ndarray,
    dayenu_filter_scale: float = 0.05, num_freq_bins: int = 500
) -> Tuple[float, float]:
    """Implements a search strategy that "straddles" each frequency bin."""
    best_snr_overall, best_z_overall = -np.inf, None
    search_freqs = np.linspace(np.min(freqs), np.max(freqs), num_freq_bins)
    
    for freq_center in search_freqs:
        z_center = freq_to_z(freq_center)
        freq_width = np.mean(np.diff(freqs))
        z_delta = np.abs(freq_to_z(freq_center + freq_width / 2) - z_center)
        z_trials = [z_center - z_delta, z_center + z_delta]
        
        best_snr_in_bin, best_z_in_bin, _ = run_pipeline_aware_matched_filter(
            spectrum, freqs, intrinsic_template_v, vel_axis_kms, rfi_weights,
            noise_per_channel, z_trials, dayenu_filter_scale
        )
        
        if best_snr_in_bin > best_snr_overall:
            best_snr_overall, best_z_overall = best_snr_in_bin, best_z_in_bin
            
    return best_snr_overall, best_z_overall
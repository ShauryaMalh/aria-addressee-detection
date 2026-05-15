"""
Direction-of-Arrival estimation using SRP-PHAT on the Aria 7-mic array.
"""
import numpy as np
import pyroomacoustics as pra


# Aria Gen 1 microphone positions in the device frame (meters).
# Source: SAVVY paper, section E.2.
# Mic 0: right-front-bottom    ( 0.05, -0.04,  0.00)
# Mic 1: bridge of nose        (-0.005, 0.00,  0.00)
# Mic 2: left-front-bottom     (-0.05, -0.04,  0.00)
# Mic 3: far-left-up           (-0.07,  0.00,  0.00)
# Mic 4: far-right-up          ( 0.07,  0.00,  0.00)
# Mic 5: rear left leg         (-0.07,  0.00, -0.10)
# Mic 6: rear right leg        ( 0.07,  0.00, -0.10)
MIC_POSITIONS_DEVICE = np.array([
    [ 0.050, -0.040,  0.000],  # 0
    [-0.005,  0.000,  0.000],  # 1
    [-0.050, -0.040,  0.000],  # 2
    [-0.070,  0.000,  0.000],  # 3
    [ 0.070,  0.000,  0.000],  # 4
    [-0.070,  0.000, -0.100],  # 5
    [ 0.070,  0.000, -0.100],  # 6
]).T  # shape (3, 7): each column is one mic's position

# Per SAVVY Table 4, mics {3, 4, 5, 6} give the best front/back disambiguation
DEFAULT_MIC_INDICES = [0, 1, 2, 3, 4, 5, 6]  
SAMPLE_RATE = 48000
NFFT = 1024


def estimate_doa(audio_window, mic_indices=None, sample_rate=SAMPLE_RATE, nfft=NFFT,
                 freq_range=(300, 3500)):
    """Estimate direction-of-arrival in the horizontal (x-z) plane.

    In the Aria device frame:
      x = right (+) / left (-)
      y = up (+) / down (-)
      z = front (+) / back (-)

    We project mic positions onto the x-z plane for 2D DoA.
    Azimuth convention (from pyroomacoustics, measured from +x axis):
      0° = wearer's right
      90° = forward (nose direction)
      ±180° = wearer's left
      -90° = behind the wearer

    Returns:
        azimuth_rad: estimated azimuth in radians, in [-pi, pi].
        spatial_spectrum: the full azimuth response.
        azimuth_grid: candidate azimuths in radians.
    """
    if mic_indices is None:
        mic_indices = DEFAULT_MIC_INDICES

    x = audio_window[:, mic_indices].astype(np.float64)
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs

    # Project mic positions onto horizontal plane: use x (row 0) and z (row 2)
    L_3d = MIC_POSITIONS_DEVICE[:, mic_indices]  # (3, M)
    L_2d = np.array([
        L_3d[0, :],  # x: left-right
        L_3d[2, :],  # z: front-back
    ])  # (2, M)

    stft = pra.transform.stft.analysis(x, nfft, nfft // 2)
    stft = stft.transpose([2, 1, 0])

    azimuth_grid = np.deg2rad(np.arange(-180, 180, 1))

    doa = pra.doa.SRP(L_2d, sample_rate, nfft,
                      azimuth=azimuth_grid,
                      num_src=1,
                      dim=2,
                      mode="far")

    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    freq_bins = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]

    doa.locate_sources(stft, freq_bins=freq_bins)

    azimuth_rad = float(doa.azimuth_recon[0])
    spatial_spectrum = doa.grid.values.copy()

    return azimuth_rad, spatial_spectrum, azimuth_grid



def classify_speaker(spatial_spectrum, azimuth_grid,
                     forward_deg=90.0,
                     forward_halfwidth_deg=20.0,
                     dominance_ratio=1.3):
    """Decide who is likely speaking based on SRP-PHAT spatial response.

    In the Aria x-z plane convention:
      +90° = forward (wearer's nose)
      0° = wearer's right
      -90° = behind the wearer
      ±180° = wearer's left

    Returns:
        label: "wearer", "other", or "ambiguous"
        forward_energy: max spectrum value in the forward cone
        off_axis_energy: max spectrum value outside the forward cone
        off_axis_az_deg: azimuth of the off-axis peak
        az_deg: the overall peak azimuth
    """
    az_deg_arr = np.rad2deg(azimuth_grid)

    # Angular distance to forward, wrapping correctly
    angular_dist = np.abs(az_deg_arr - forward_deg)
    angular_dist = np.minimum(angular_dist, 360 - angular_dist)

    forward_mask = angular_dist < forward_halfwidth_deg

    forward_energy = spatial_spectrum[forward_mask].max() if forward_mask.any() else 0.0

    off_axis_vals = spatial_spectrum[~forward_mask]
    off_axis_energy = off_axis_vals.max() if len(off_axis_vals) > 0 else 0.0
    off_axis_idx = np.argmax(spatial_spectrum * (~forward_mask).astype(float))
    off_axis_az_deg = float(az_deg_arr[off_axis_idx])

    # Overall peak
    peak_idx = np.argmax(spatial_spectrum)
    peak_az_deg = float(az_deg_arr[peak_idx])

    # Decision
    if forward_mask[peak_idx]:
        # Peak is in the forward cone
        if forward_energy > dominance_ratio * off_axis_energy:
            label = "wearer"
        else:
            label = "ambiguous"
    else:
        # Peak is outside the forward cone
        if off_axis_energy > dominance_ratio * forward_energy:
            label = "other"
        else:
            label = "ambiguous"

    return label, forward_energy, off_axis_energy, off_axis_az_deg, peak_az_deg



if __name__ == "__main__":
    from perception import load_sequence, get_context, cluster_utterances

    print("Loading sequence...")
    h = load_sequence()

    print("Clustering utterances...")
    utterances = cluster_utterances(h["speech"])
    print(f"  Found {len(utterances)} utterances\n")

    for i, utt in enumerate(utterances):
        ts = utt["mid_ns"]
        ctx = get_context(h, ts)

        if ctx["pose"] is None:
            print(f"  [{i}] t={ts/1e9:.2f}s: OUT OF RANGE")
            continue

        if ctx["audio_window"] is None or len(ctx["audio_window"]) < 1000:
            print(f"  [{i}] t={ts/1e9:.2f}s: no audio")
            continue

        az_rad, spectrum, grid = estimate_doa(ctx["audio_window"])
        label, fwd_e, off_e, off_az, peak_az = classify_speaker(spectrum, grid)

        gaze_info = f"gaze yaw={ctx['gaze'].yaw:.2f}" if ctx["gaze"] else ""

        print(f"  [{i}] t={ts/1e9:.2f}s — \"{utt['text'][:50]}{'...' if len(utt['text']) > 50 else ''}\"")
        print(f"      DoA peak: {peak_az:+6.1f}°  |  speaker: {label}")
        print(f"      Forward energy: {fwd_e:.3f}  |  Off-axis: {off_e:.3f} @ {off_az:+.0f}°")
        if gaze_info:
            print(f"      {gaze_info}")
        print()
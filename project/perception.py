"""
Perception layer for AEA addressee detection project.
Loads all data sources for one sequence using the official MPS loaders.
"""
from pathlib import Path
import pandas as pd
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider
from projectaria_tools.core.mps.utils import (
    get_nearest_pose,
    get_nearest_eye_gaze,
)



"""
To download dataset:
aria_dataset_downloader \
aria_dataset_downloader \
  -c /homes/iws/shaurm/projectaria_sandbox/aea_download_urls.json \
  -o /homes/iws/shaurm/projectaria_sandbox/aea_data \
  -l loc4_script2_seq4_rec1
"""

SEQ_DIR = Path("/homes/iws/shaurm/projectaria_sandbox/aea_data/loc2_script2_seq4_rec2")



def cluster_utterances(speech_df, gap_ns=1_500_000_000):
    """Group word-level transcript rows into utterance-level clusters.
    
    Words separated by less than gap_ns are considered part of the same utterance.
    
    Returns:
        list of dicts, each with:
            - start_ns, end_ns, mid_ns: timestamps
            - text: concatenated words
            - num_words: word count
            - words_df: the original DataFrame slice
    """
    if len(speech_df) == 0:
        return []
    
    utterances = []
    current_words = [speech_df.iloc[0]]
    
    for i in range(1, len(speech_df)):
        prev_end = speech_df.iloc[i-1]["endTime_ns"]
        curr_start = speech_df.iloc[i]["startTime_ns"]
        
        if curr_start - prev_end < gap_ns:
            current_words.append(speech_df.iloc[i])
        else:
            words_df = pd.DataFrame(current_words)
            text = " ".join(w["written"] for w in current_words)
            utterances.append({
                "start_ns": int(current_words[0]["startTime_ns"]),
                "end_ns": int(current_words[-1]["endTime_ns"]),
                "mid_ns": int((current_words[0]["startTime_ns"] + current_words[-1]["endTime_ns"]) / 2),
                "text": text,
                "num_words": len(current_words),
                "words_df": words_df,
            })
            current_words = [speech_df.iloc[i]]
    
    # Last cluster
    words_df = pd.DataFrame(current_words)
    text = " ".join(w["written"] for w in current_words)
    utterances.append({
        "start_ns": int(current_words[0]["startTime_ns"]),
        "end_ns": int(current_words[-1]["endTime_ns"]),
        "mid_ns": int((current_words[0]["startTime_ns"] + current_words[-1]["endTime_ns"]) / 2),
        "text": text,
        "num_words": len(current_words),
        "words_df": words_df,
    })
    
    # Filter out single filler words
    utterances = [u for u in utterances if u["num_words"] >= 2]
    
    return utterances

def get_context(handles, query_ts_ns):
    """Pull everything we need for a single utterance at timestamp query_ts_ns."""
    import numpy as np
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

    # --- Pose (1 kHz, dense) ---
    pose_info = get_nearest_pose(handles["trajectory"], query_ts_ns)
    if pose_info is None:
        T_world_device = None
        position_world = None
    else:
        T_world_device = pose_info.transform_world_device
        position_world = T_world_device.translation().flatten()  # (3,) array

    # --- Gaze (10 Hz, sparse) ---
    gaze = get_nearest_eye_gaze(handles["gaze"], query_ts_ns)

    # --- Speech: grab words within ±3 seconds of query_ts_ns ---
    window_ns = 3_000_000_000  # 3s window
    speech = handles["speech"]
    mask = (speech["startTime_ns"] >= query_ts_ns - window_ns) & \
           (speech["endTime_ns"] <= query_ts_ns + window_ns)
    speech_words = speech[mask].reset_index(drop=True)

    # --- RGB frame nearest to query_ts_ns ---
    rgb_stream_id = StreamId("214-1")
    provider = handles["provider"]

    frame_index = provider.get_index_by_time_ns(
        rgb_stream_id, query_ts_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
    )
    rgb_data = provider.get_image_data_by_index(rgb_stream_id, frame_index)
    rgb_array = rgb_data[0].to_numpy_array()  # (H, W, 3) uint8
    rgb_ts_ns = rgb_data[1].capture_timestamp_ns


    # --- Audio: 7-channel window centered on query_ts_ns ---
    mic_stream_id = StreamId("231-1")
    audio_window_s = 2.0  # 2-second window
    half_window_ns = int(audio_window_s * 1e9 / 2)

    # Find audio data blocks covering our window
    start_ts = query_ts_ns - half_window_ns
    end_ts = query_ts_ns + half_window_ns

    # The mic stream returns chunks; we need to collect samples across chunks
    # that fall within our window
    num_audio_records = provider.get_num_data(mic_stream_id)
    audio_samples = []  # list of (timestamp_ns, samples) tuples
    for i in range(num_audio_records):
        data = provider.get_audio_data_by_index(mic_stream_id, i)
        audio_block = data[0]  # AudioData
        meta = data[1]          # AudioDataRecord
        # meta.capture_timestamps_ns is a list of per-sample timestamps
        block_ts = meta.capture_timestamps_ns
        if len(block_ts) == 0:
            continue
        if block_ts[-1] < start_ts:
            continue
        if block_ts[0] > end_ts:
            break
        audio_samples.append((block_ts, audio_block.data))

    # Concatenate blocks and slice to exact window
    # Concatenate blocks and slice to exact window
    if audio_samples:
        import numpy as np
        all_ts = np.concatenate([np.asarray(ts) for ts, _ in audio_samples])

        # Each block's samples come back as a flat 1D array of length (n_frames * 7),
        # interleaved as [ch0_s0, ch1_s0, ..., ch6_s0, ch0_s1, ...].
        # Reshape each block to (n_frames, 7) before concatenating.
        reshaped_blocks = []
        for ts, samples in audio_samples:
            n_frames = len(ts)
            arr = np.asarray(samples).reshape(n_frames, 7)
            reshaped_blocks.append(arr)
        all_samples = np.concatenate(reshaped_blocks, axis=0)  # (N_frames, 7)

        # Mask on frames (aligned with all_ts)
        in_window = (all_ts >= start_ts) & (all_ts <= end_ts)
        audio_window = all_samples[in_window]      # (N, 7)
        audio_window_ts = all_ts[in_window]
    else:
        audio_window = None
        audio_window_ts = None
    return {
        "query_ts_ns": query_ts_ns,
        "pose": T_world_device,
        "position_world": position_world,
        "gaze": gaze,
        "speech_words": speech_words,
        "rgb_frame": rgb_array,
        "rgb_timestamp_ns": rgb_ts_ns,
        "audio_window": audio_window,           # (N, 7) multichannel float array
        "audio_window_ts": audio_window_ts,     # (N,) per-sample timestamps
    }


def load_sequence(seq_dir: Path = SEQ_DIR):
    """Load every data source for a sequence.

    Returns a dict with:
      - provider: VRS data provider (for frames, audio)
      - speech: pandas DataFrame of word-level transcript
      - trajectory: list of ClosedLoopPose objects (from MPS)
      - gaze: list of EyeGaze objects (from MPS)
      - points: list of GlobalPointPosition objects (point cloud)
    """
    vrs_path = seq_dir / "recording.vrs"
    provider = data_provider.create_vrs_data_provider(str(vrs_path))

    speech = pd.read_csv(seq_dir / "speech.csv")


    #Returns a list of ClosedLoopPose objects - each one has typed attributes
    trajectory = mps.read_closed_loop_trajectory(
        str(seq_dir / "mps" / "slam" / "closed_loop_trajectory.csv")
    )

    # returns a list of EyeGaze objects.
    gaze = mps.read_eyegaze(
        str(seq_dir / "mps" / "eye_gaze" / "general_eye_gaze.csv")
    )

    #Returns a list of GlobalPointPosition objects. 
    points = mps.read_global_point_cloud(
        str(seq_dir / "mps" / "slam" / "semidense_points.csv.gz")
    )

    return {
        "provider": provider,
        "speech": speech,
        "trajectory": trajectory,
        "gaze": gaze,
        "points": points,
    }


if __name__ == "__main__":
    print("Loading sequence...")
    h = load_sequence()
    print(f"  Loaded: {len(h['trajectory'])} poses, {len(h['gaze'])} gaze, {len(h['speech'])} speech words\n")

    # Auto-detect utterances instead of hardcoding timestamps
    utterances = cluster_utterances(h["speech"])
    print(f"  Found {len(utterances)} utterances:\n")
    for i, utt in enumerate(utterances):
        print(f"    [{i}] {utt['start_ns']/1e9:.2f}s - {utt['end_ns']/1e9:.2f}s: {utt['text'][:60]}")

    # Test get_context on the first utterance
    if utterances:
        test_utt = utterances[0]
        test_ts = test_utt["mid_ns"]
        print(f"\n  Testing get_context on utterance [0] at t={test_ts/1e9:.2f}s...")
        ctx = get_context(h, test_ts)

        print(f"  Wearer position: {ctx['position_world']}")
        print(f"  Gaze yaw/pitch: {ctx['gaze'].yaw:.3f}, {ctx['gaze'].pitch:.3f}" if ctx['gaze'] else "  No gaze")
        print(f"  RGB frame shape: {ctx['rgb_frame'].shape}")
        print(f"  Speech words in window: {len(ctx['speech_words'])}")
        print(f"  Audio window shape: {ctx['audio_window'].shape if ctx['audio_window'] is not None else 'None'}")
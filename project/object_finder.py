"""
Find real object positions in the world frame using GroundingDINO.
Unlike YOLO (fixed 80-class vocabulary), GroundingDINO is open-vocabulary:
you give it a text query ("creamer", "spoon") and it finds matching objects.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.mps.utils import get_nearest_pose


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_GDINO_MODEL = None
_GDINO_PROCESSOR = None


def _load_model():
    global _GDINO_MODEL, _GDINO_PROCESSOR
    if _GDINO_MODEL is None:
        print("Loading GroundingDINO model...")
        _GDINO_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).to(_DEVICE)
        _GDINO_PROCESSOR = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        )
        _GDINO_MODEL.eval()
        print(f"  GroundingDINO loaded on {_DEVICE}")

def clean_label(label):
    """Remove BERT tokenizer artifacts from GroundingDINO labels."""
    # Remove ## markers and join to previous fragment
    label = label.replace("##", "")
    # Remove extra spaces from the join
    label = " ".join(label.split())
    # Take only the first meaningful word if there are duplicates
    words = label.split()
    seen = set()
    unique = []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)
    return " ".join(unique).strip()
    


def detect_objects(rgb_frame, text_query, confidence_threshold=0.25):
    """Detect objects matching a text query in an RGB frame.

    Args:
        rgb_frame: numpy array (H, W, 3) uint8
        text_query: string like "creamer. spoon. person." 
                    (period-separated list of things to find)
        confidence_threshold: minimum confidence to keep a detection

    Returns:
        list of dicts, each with:
            - label: matched text label
            - confidence: float
            - bbox: [x1, y1, x2, y2] in pixels
            - center_px: (cx, cy) pixel center
    """
    _load_model()

    # Convert to PIL image
    image = Image.fromarray(rgb_frame)
    
    if not text_query.endswith("."):
        text_query = text_query + "."
    
    inputs = _GDINO_PROCESSOR(
        images=image, 
        text=text_query, 
        return_tensors="pt"
    ).to(_DEVICE)

    with torch.no_grad():
        outputs = _GDINO_MODEL(**inputs)

    # Get raw predictions and filter manually (API-version-safe)
    target_sizes = torch.tensor([rgb_frame.shape[:2]], device=_DEVICE)
    
    try:
        # Try newer API first
        results = _GDINO_PROCESSOR.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=target_sizes
        )[0]
    except TypeError:
        # Fall back to even simpler approach
        results = _GDINO_PROCESSOR.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            target_sizes=target_sizes
        )[0]

    detections = []
    for i in range(len(results["scores"])):
        score = float(results["scores"][i].cpu())
        if score < confidence_threshold:
            continue
            
        box = results["boxes"][i].cpu().numpy().tolist()
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        label = results["labels"][i] if "labels" in results else results["text_labels"][i] if "text_labels" in results else f"object_{i}"
        if not isinstance(label, str):
            label = str(label)
        label = clean_label(label)

        detections.append({
            "label": label.strip(),
            "confidence": score,
            "bbox": [x1, y1, x2, y2],
            "center_px": (cx, cy),
        })

    return detections


def detect_in_frame_at_timestamp(handles, timestamp_ns, text_query, 
                                  confidence_threshold=0.25):

    """Run detection on the RGB frame closest to a given timestamp.

    Returns list of detections, each with an added 'timestamp_ns' field.
    """
    provider = handles["provider"]
    rgb_stream_id = StreamId("214-1")

    frame_index = provider.get_index_by_time_ns(
        rgb_stream_id, timestamp_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
    )
    rgb_data = provider.get_image_data_by_index(rgb_stream_id, frame_index)
    frame = rgb_data[0].to_numpy_array()
    ts_ns = rgb_data[1].capture_timestamp_ns

    detections = detect_objects(frame, text_query, confidence_threshold)
    
    # Add timestamp to each detection
    for d in detections:
        d["timestamp_ns"] = ts_ns
        d["frame_index"] = int(frame_index)

    return detections


def scan_for_objects(handles, text_query, num_frames=30, 
                     confidence_threshold=0.25):
    """Scan across the sequence for objects matching a text query.

    Args:
        handles: output of load_sequence()
        text_query: period-separated query like "creamer. spoon. person."
        num_frames: how many frames to sample
        box_threshold: minimum box confidence
        text_threshold: minimum text-match confidence

    Returns:
        list of detections across all sampled frames
    """
    provider = handles["provider"]
    rgb_stream_id = StreamId("214-1")
    num_total = provider.get_num_data(rgb_stream_id)
    indices = np.linspace(0, num_total - 1, num_frames, dtype=int)

    all_detections = []
    for idx in indices:
        rgb_data = provider.get_image_data_by_index(rgb_stream_id, int(idx))
        frame = rgb_data[0].to_numpy_array()
        ts_ns = rgb_data[1].capture_timestamp_ns

        detections = detect_objects(frame, text_query, confidence_threshold)
        
        for d in detections:
            d["timestamp_ns"] = ts_ns
            d["frame_index"] = int(idx)
        
        all_detections.extend(detections)

    # Report findings
    from collections import Counter
    counts = Counter(d["label"] for d in all_detections)
    print(f"  Found {len(all_detections)} detections across {num_frames} frames for query \"{text_query}\":")
    for label, count in counts.most_common():
        best = max((d for d in all_detections if d["label"] == label), key=lambda d: d["confidence"])
        print(f"    \"{label}\": {count} detections, best confidence {best['confidence']:.2f}")

    return all_detections


def project_detection_to_world(detection, handles, point_cloud_xyz):
    """Project a 2D detection to a 3D world position.

    Casts a ray from the camera through the detection center pixel,
    finds where it intersects the point cloud, and returns that 3D position.
    """
    ts_ns = detection["timestamp_ns"]
    provider = handles["provider"]

    pose = get_nearest_pose(handles["trajectory"], ts_ns)
    if pose is None:
        return None

    T_world_device = pose.transform_world_device

    rgb_stream_id = StreamId("214-1")
    rgb_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calib = provider.get_device_calibration()
    camera_calib = device_calib.get_camera_calib(rgb_label)

    T_device_camera = camera_calib.get_transform_device_camera()
    T_world_camera = T_world_device @ T_device_camera

    camera_pos_world = T_world_camera.translation().flatten()

    cx, cy = detection["center_px"]
    pixel = np.array([cx, cy])
    try:
        ray_camera = camera_calib.unproject(pixel)
        if ray_camera is None:
            return None
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
    except Exception:
        fx = fy = 1408 / (2 * np.tan(np.radians(55)))
        cx_intrinsic = cy_intrinsic = 1408 / 2
        ray_camera = np.array([
            (cx - cx_intrinsic) / fx,
            (cy - cy_intrinsic) / fy,
            1.0
        ])
        ray_camera = ray_camera / np.linalg.norm(ray_camera)

    R_world_camera = T_world_camera.rotation().to_matrix()
    ray_world = R_world_camera @ ray_camera

    pc = point_cloud_xyz
    to_points = pc - camera_pos_world
    projections = np.dot(to_points, ray_world)
    forward_mask = projections > 0.5

    if not forward_mask.any():
        return camera_pos_world + 2.0 * ray_world

    closest_on_ray = camera_pos_world + projections[:, None] * ray_world
    distances = np.linalg.norm(pc - closest_on_ray, axis=1)

    valid = forward_mask & (distances < 0.5)

    if valid.sum() < 3:
        median_dist = np.median(projections[forward_mask])
        return camera_pos_world + median_dist * ray_world

    object_pos = np.median(pc[valid], axis=0)
    return object_pos


def find_object_world_position(handles, point_cloud_xyz, text_query, num_frames=30):
    """Find an object's world position by scanning frames with a text query.

    Args:
        handles: output of load_sequence()
        point_cloud_xyz: filtered point cloud
        text_query: what to search for (e.g., "creamer" or "spoon")
        num_frames: how many frames to scan

    Returns:
        world_pos: (3,) array or None
        best_detection: the detection dict, or None
    """
    detections = scan_for_objects(handles, text_query, num_frames=num_frames)

    if not detections:
        print(f"  \"{text_query}\" not found in any frame.")
        return None, None

    best = max(detections, key=lambda d: d["confidence"])
    print(f"\n  Using best detection: frame {best['frame_index']}, "
          f"conf {best['confidence']:.2f}, label \"{best['label']}\", "
          f"center px {best['center_px']}")

    world_pos = project_detection_to_world(best, handles, point_cloud_xyz)
    if world_pos is not None:
        print(f"  Projected to world position: ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})")

    return world_pos, best


if __name__ == "__main__":
    from perception import load_sequence
    from visualization import load_filtered_points

    print("\nLoading sequence...")
    h = load_sequence()

    print("\nFiltering point cloud...")
    pc_xyz = load_filtered_points(h)
    print(f"  {len(pc_xyz)} points")

    # Test open-vocabulary detection — search for specific objects from the transcript
    test_queries = [
        "creamer. cream. bottle.",
        "spoon. utensil.",
        "person. human.",
        "refrigerator. fridge.",
        "popcorn. snack bag.",
    ]

    for query in test_queries:
        print(f"\n--- Searching for: \"{query}\" ---")
        pos, det = find_object_world_position(h, pc_xyz, query, num_frames=20)
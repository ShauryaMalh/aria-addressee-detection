"""
Method A: Structured pipeline for addressee detection + task extraction.
Combines DoA, head pose, gaze, person detection, and object grounding.
"""
import numpy as np
from perception import load_sequence, get_context, cluster_utterances
from doa import estimate_doa, classify_speaker


# ============================================================
# Commonsense fallback: object → likely location
# Used ONLY when gaze-based grounding fails
# ============================================================
OBJECT_LOCATION_MAP = {
    # Food / drink items → refrigerator
    "creamer": "refrigerator", "cream": "refrigerator",
    "milk": "refrigerator", "butter": "refrigerator",
    "eggs": "refrigerator", "cheese": "refrigerator",
    "water": "refrigerator", "juice": "refrigerator",
    "soda": "refrigerator", "beer": "refrigerator",
    "yogurt": "refrigerator",
    # Utensils → general kitchen area
    "spoon": "dining table", "fork": "dining table",
    "knife": "dining table", "plate": "dining table",
    "cup": "dining table", "bowl": "dining table",
    "glass": "dining table",
    # Snacks
    "popcorn": "microwave", "snack": "dining table",
    # Misc
    "coffee": "dining table", "towel": "oven",
}


def compute_gaze_direction_world(ctx):
    """Convert gaze (yaw, pitch in device/CPF frame) to a world-frame direction vector."""
    gaze = ctx["gaze"]
    if gaze is None:
        return None

    # Convert from yaw/pitch to a 3D unit vector in device frame, spherical-to-Cartesian conversion. 
    yaw = gaze.yaw
    pitch = gaze.pitch
    gaze_device = np.array([
        np.sin(yaw) * np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw) * np.cos(pitch),
    ])

    T_world_device = ctx["pose"]
    if T_world_device is None:
        return None

    R = T_world_device.rotation().to_matrix()
    gaze_world = R @ gaze_device
    return gaze_world


def compute_forward_direction_world(ctx):
    """Get the wearer's forward direction (nose direction) in world frame."""
    if ctx["pose"] is None:
        return None
    R = ctx["pose"].rotation().to_matrix()
    forward_device = np.array([0, 0, 1.0])
    forward_world = R @ forward_device
    return forward_world


def extract_task_from_words(speech_words):
    """Rule-based task extraction from transcript words.

    Returns:
        has_task: bool
        task_desc: full text of the utterance window, or None
        has_deictic: bool
        mentioned_object: the first object noun found, or None
    """
    words = speech_words["written"].str.lower().str.strip(".,!?").tolist()
    text = " ".join(words)

    action_verbs = ["get", "grab", "hand", "put", "bring", "give", "pass",
                    "take", "move", "stir", "pour", "open", "close", "set"]

    deictics = ["that", "this", "it", "these", "those", "there", "here"]

    # Object nouns we can try to detect or look up
    object_nouns = list(OBJECT_LOCATION_MAP.keys()) + [
        "fridge", "refrigerator", "microwave", "oven", "table",
        "chair", "counter", "cabinet", "sink"
    ]

    found_verbs = [v for v in action_verbs if v in words]
    found_deictics = [d for d in deictics if d in words]
    has_task = len(found_verbs) > 0
    has_deictic = len(found_deictics) > 0

    # Find the first mentioned object
    mentioned_object = None
    for obj in object_nouns:
        if obj in words:
            mentioned_object = obj
            break

    task_desc = None
    if has_task:
        task_desc = text

    return has_task, task_desc, has_deictic, mentioned_object


def ground_task_target(ctx, mentioned_object, handles, point_cloud_xyz, utterances=None):
    
    """
    Ground a verbal task to a physical location.
    Strategy (in priority order):
      0. DEICTIC: If only a deictic ("that"/"this"), find what the wearer is looking at
      1. DIRECT: Search the current frame for the mentioned object by name
      2. GAZE: Find the object closest to the wearer's gaze direction
      3. LOOKUP: Commonsense mapping + broader scene scan
    """
    from object_finder import detect_objects, project_detection_to_world, find_object_world_position

    wearer_pos = ctx["position_world"]
    gaze_world = compute_gaze_direction_world(ctx)
    frame = ctx["rgb_frame"]

    # --- Strategy 0: Deictic resolution ---
    if mentioned_object == "__deictic__":
        # Step A: Find objects mentioned earlier in the conversation
        context_objects = []
        context_nouns = list(OBJECT_LOCATION_MAP.keys()) + [
            "fridge", "refrigerator", "microwave", "oven", "table",
            "cup", "bottle", "bag", "box"
        ]
        
        words = ctx["speech_words"]["written"].str.lower().str.strip(".,!?").tolist()
        for noun in context_nouns:
            if noun in words:
                context_objects.append(noun)
        
        if utterances is not None:
            current_time = ctx["query_ts_ns"]
            for utt in utterances:
                if utt["mid_ns"] < current_time:
                    utt_words = utt["text"].lower().split()
                    for noun in context_nouns:
                        if noun in utt_words and noun not in context_objects:
                            context_objects.append(noun)

        # Step B: FIRST try to find context objects specifically
        # If "creamer" was discussed earlier, search for creamer first
        if context_objects and gaze_world is not None:
            context_query = ". ".join(context_objects) + "."
            context_detections = detect_objects(frame, context_query, confidence_threshold=0.2)
            print(f"    Deictic context search: query=\"{context_query}\", found {len(context_detections)} detections")
            for cd in context_detections:
                print(f"      \"{cd['label']}\" conf={cd['confidence']:.2f}")
            
            if context_detections:
                best_det = None
                best_alignment = -1
                best_world_pos = None
                best_dist = 0

                for det in context_detections:
                    det["timestamp_ns"] = ctx["query_ts_ns"]
                    det_world = project_detection_to_world(det, handles, point_cloud_xyz)
                    if det_world is None:
                        continue

                    to_det = det_world - wearer_pos
                    dist = np.linalg.norm(to_det)
                    if dist < 0.2 or dist > 5.0:
                        continue

                    to_det_norm = to_det / (dist + 1e-8)
                    alignment = float(np.dot(gaze_world, to_det_norm))

                    if alignment > best_alignment:
                        best_alignment = alignment
                        best_det = det
                        best_world_pos = det_world
                        best_dist = dist

                if best_det is not None and best_alignment > 0.4:
                    # Match the detection label to the context object name
                    display_label = best_det["label"]
                    for co in context_objects:
                        if co in best_det["label"].lower() or best_det["label"].lower() in co:
                            display_label = co
                            break
                    
                    reasoning = (
                        f"Deictic resolution: wearer said 'that'/'this'. "
                        f"Conversation previously mentioned: {context_objects}. "
                        f"Found \"{display_label}\" in current frame "
                        f"(alignment: {best_alignment:.2f}, distance: {best_dist:.1f}m)."
                    )
                    return best_world_pos, display_label, "deictic", reasoning

        # Step C: If no context match, fall back to general object search
        if gaze_world is not None:
            general_query = "bottle. cup. bowl. plate. spoon. container. bag. box. food."
            general_detections = detect_objects(frame, general_query, confidence_threshold=0.2)

            if general_detections:
                best_det = None
                best_alignment = -1
                best_world_pos = None
                best_dist = 0

                for det in general_detections:
                    det["timestamp_ns"] = ctx["query_ts_ns"]
                    det_world = project_detection_to_world(det, handles, point_cloud_xyz)
                    if det_world is None:
                        continue

                    to_det = det_world - wearer_pos
                    dist = np.linalg.norm(to_det)
                    if dist < 0.2 or dist > 5.0:
                        continue

                    to_det_norm = to_det / (dist + 1e-8)
                    alignment = float(np.dot(gaze_world, to_det_norm))

                    if alignment > best_alignment:
                        best_alignment = alignment
                        best_det = det
                        best_world_pos = det_world
                        best_dist = dist

                if best_det is not None and best_alignment > 0.5:
                    reasoning = (
                        f"Deictic resolution (general): wearer said 'that'/'this'. "
                        f"Gaze toward \"{best_det['label']}\" "
                        f"(alignment: {best_alignment:.2f}, distance: {best_dist:.1f}m). "
                        f"No conversation context matched."
                    )
                    return best_world_pos, best_det["label"], "deictic", reasoning

        align_str = "N/A"
        reasoning = (
            f"Deictic resolution failed. Context: {context_objects}."
        )
        return None, None, "failed", reasoning

    # --- Strategy 1: Direct search for the mentioned object ---
    if mentioned_object:
        synonyms = {
            "creamer": "creamer. cream. cream bottle.",
            "cream": "cream. creamer. cream bottle.",
            "spoon": "spoon. utensil.",
            "milk": "milk. milk carton. milk bottle.",
            "popcorn": "popcorn. popcorn bag. snack bag.",
            "coffee": "coffee. coffee cup. coffee mug.",
            "cup": "cup. mug. glass.",
            "bowl": "bowl.",
            "plate": "plate. dish.",
            "eggs": "eggs. egg carton.",
        }
        query = synonyms.get(mentioned_object, f"{mentioned_object}.")

        detections = detect_objects(frame, query, confidence_threshold=0.25)

        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            best["timestamp_ns"] = ctx["query_ts_ns"]
            det_world = project_detection_to_world(best, handles, point_cloud_xyz)

            if det_world is not None:
                reasoning = (
                    f"Direct visual grounding: searched for \"{mentioned_object}\" "
                    f"→ detected (confidence: {best['confidence']:.2f}) "
                    f"at ({det_world[0]:.1f}, {det_world[1]:.1f}, {det_world[2]:.1f})."
                )
                return det_world, mentioned_object, "direct", reasoning

        print(f"    Direct search: \"{mentioned_object}\" not in current frame, scanning...")
        world_pos, best_det = find_object_world_position(
            handles, point_cloud_xyz, query, num_frames=10
        )
        if world_pos is not None:
            reasoning = (
                f"Direct visual grounding (scan): \"{mentioned_object}\" "
                f"→ found \"{best_det['label']}\" (confidence: {best_det['confidence']:.2f}) "
                f"at ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f})."
            )
            return world_pos, mentioned_object, "direct", reasoning

    # --- Strategy 2: Gaze-based grounding ---
    if gaze_world is not None:
        general_detections = detect_objects(frame, "object. item. container. appliance.",
                                            confidence_threshold=0.25)

        if general_detections:
            best_det = None
            best_alignment = -1
            best_world_pos = None

            for det in general_detections:
                det["timestamp_ns"] = ctx["query_ts_ns"]
                det_world = project_detection_to_world(det, handles, point_cloud_xyz)
                if det_world is None:
                    continue

                to_det = det_world - wearer_pos
                dist = np.linalg.norm(to_det)
                if dist < 0.3:
                    continue

                to_det_norm = to_det / (dist + 1e-8)
                alignment = float(np.dot(gaze_world, to_det_norm))

                if alignment > best_alignment:
                    best_alignment = alignment
                    best_det = det
                    best_world_pos = det_world

            if best_det is not None and best_alignment > 0.7:
                reasoning = (
                    f"Gaze-grounded: wearer looking toward \"{best_det['label']}\" "
                    f"(alignment: {best_alignment:.2f}, "
                    f"distance: {np.linalg.norm(best_world_pos - wearer_pos):.1f}m)."
                )
                return best_world_pos, best_det["label"], "gaze", reasoning

    # --- Strategy 3: Lookup fallback ---
    if mentioned_object and mentioned_object in OBJECT_LOCATION_MAP:
        target_location_type = OBJECT_LOCATION_MAP[mentioned_object]
        query = f"{target_location_type}."

        world_pos, best_det = find_object_world_position(
            handles, point_cloud_xyz, query, num_frames=10
        )
        if world_pos is not None:
            reasoning = (
                f"Lookup fallback: \"{mentioned_object}\" → {target_location_type}. "
                f"Found \"{best_det['label']}\" "
                f"at ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f})."
            )
            return world_pos, f"{target_location_type} (via {mentioned_object})", "lookup", reasoning

    # --- All failed ---
    reasoning = f"Could not ground task. Mentioned: \"{mentioned_object or 'none'}\"."
    return None, None, "failed", reasoning



def method_a_decide(ctx, handles=None, point_cloud_xyz=None, 
                    assumed_robot_position_world=None, utterances=None):
    """Full Method A decision pipeline.

    Args:
        ctx: output of get_context() for this timestamp.
        handles: output of load_sequence() (needed for object grounding).
        point_cloud_xyz: filtered point cloud array (needed for object grounding).
        assumed_robot_position_world: (3,) array of the robot's assumed position.

    Returns:
        decision: dict with addressee, task, target, reasoning trace.
    """
    reasoning = {}

    # --- Step 1: DoA-based speaker classification ---
    if ctx["audio_window"] is not None and len(ctx["audio_window"]) > 1000:
        az_rad, spectrum, grid = estimate_doa(ctx["audio_window"])
        label, fwd_e, off_e, off_az_deg, peak_az_deg = classify_speaker(
            spectrum, grid, dominance_ratio=1.05
        )
        reasoning["doa_peak_deg"] = peak_az_deg
        reasoning["doa_speaker"] = label
        reasoning["doa_forward_energy"] = fwd_e
        reasoning["doa_offaxis_energy"] = off_e
        reasoning["doa_offaxis_az_deg"] = off_az_deg
    else:
        label = "unknown"
        reasoning["doa_speaker"] = "no_audio"

    # --- Step 2: Gaze direction in world frame ---
    gaze_world = compute_gaze_direction_world(ctx)
    forward_world = compute_forward_direction_world(ctx)
    reasoning["gaze_world"] = gaze_world.tolist() if gaze_world is not None else None
    reasoning["forward_world"] = forward_world.tolist() if forward_world is not None else None

    # --- Step 3: Gaze-to-robot alignment ---
    if gaze_world is not None and ctx["position_world"] is not None:
        if assumed_robot_position_world is None:
            assumed_robot_position_world = ctx["position_world"] + 2.0 * forward_world

        wearer_to_robot = assumed_robot_position_world - ctx["position_world"]
        wearer_to_robot_norm = wearer_to_robot / (np.linalg.norm(wearer_to_robot) + 1e-8)

        gaze_robot_cos = float(np.dot(gaze_world, wearer_to_robot_norm))
        reasoning["gaze_robot_alignment"] = gaze_robot_cos

        gaze_forward_cos = float(np.dot(gaze_world, forward_world))
        reasoning["gaze_forward_alignment"] = gaze_forward_cos
    else:
        gaze_robot_cos = 0.0
        reasoning["gaze_robot_alignment"] = None

    # --- Step 4: Task extraction from transcript ---
    has_task, task_desc, has_deictic, mentioned_object = extract_task_from_words(ctx["speech_words"])
    reasoning["has_task"] = has_task
    reasoning["task_description"] = task_desc
    reasoning["has_deictic"] = has_deictic
    reasoning["mentioned_object"] = mentioned_object

    # --- Step 4b: Ground the task to a physical location ---
    target_pos = None
    target_label = None
    grounding_method = None
    grounding_reasoning = None

    if has_task and handles is not None and point_cloud_xyz is not None:
        search_object = mentioned_object
        if search_object is None and has_deictic:
            search_object = "__deictic__"

        target_pos, target_label, grounding_method, grounding_reasoning = ground_task_target(
            ctx, search_object, handles, point_cloud_xyz, utterances=utterances
        )
        reasoning["target_pos"] = target_pos.tolist() if target_pos is not None else None
        reasoning["target_label"] = target_label
        reasoning["grounding_method"] = grounding_method
        reasoning["grounding"] = grounding_reasoning

    # --- Step 5: Decision logic ---
    doa_says_other = (label == "other")
    doa_says_wearer = (label == "wearer")
    gaze_toward_robot = gaze_robot_cos > 0.7

    if doa_says_other:
        addressee = "other_human"
        should_act = False
    elif doa_says_wearer and has_task and gaze_toward_robot:
        addressee = "robot"
        should_act = True
    elif doa_says_wearer and has_task and not gaze_toward_robot:
        addressee = "other_human"
        should_act = False
    elif doa_says_wearer and not has_task:
        addressee = "no_one"
        should_act = False
    else:
        if has_task and gaze_toward_robot:
            addressee = "robot"
            should_act = True
        elif has_task:
            addressee = "other_human"
            should_act = False
        else:
            addressee = "no_one"
            should_act = False

    # --- Step 6: Build output ---
    decision = {
        "addressed": should_act,
        "addressee": addressee,
        "task": task_desc if should_act else None,
        "has_deictic": has_deictic if should_act else False,
        "mentioned_object": mentioned_object,
        "target_pos": target_pos if should_act else None,
        "target_label": target_label if should_act else None,
        "grounding_method": grounding_method if should_act else None,
        "grounding": grounding_reasoning if should_act else None,
        "reasoning": reasoning,
    }

    return decision


# Demo: run on all detected utterances (no hardcoded timestamps)
if __name__ == "__main__":
    from visualization import load_filtered_points

    print("Loading sequence...")
    h = load_sequence()

    print("Filtering point cloud...")
    pc_xyz = load_filtered_points(h)
    print(f"  {len(pc_xyz)} points\n")

    # Auto-detect all utterances
    print("Clustering utterances...")
    utterances = cluster_utterances(h["speech"])
    print(f"  Found {len(utterances)} utterances:\n")

    for i, utt in enumerate(utterances):
        print(f"  [{i}] {utt['start_ns']/1e9:.2f}s - {utt['end_ns']/1e9:.2f}s  "
              f"({utt['num_words']} words): {utt['text'][:60]}{'...' if len(utt['text']) > 60 else ''}")

    print(f"\n{'='*70}")
    print("Running Method A on each utterance...\n")

    for i, utt in enumerate(utterances):
        ctx = get_context(h, utt["mid_ns"])
        if ctx["pose"] is None:
            print(f"  [{i}] t={utt['mid_ns']/1e9:.2f}s: OUT OF RANGE, skipping\n")
            continue

        decision = method_a_decide(ctx, handles=h, point_cloud_xyz=pc_xyz, utterances=utterances)

        r = decision["reasoning"]
        print(f"  [{i}] t={utt['mid_ns']/1e9:.2f}s — \"{utt['text'][:50]}{'...' if len(utt['text']) > 50 else ''}\"")
        print(f"      DoA: {r.get('doa_peak_deg', '?'):+.0f}° ({r.get('doa_speaker', '?')})")
        print(f"      Gaze→robot: {r.get('gaze_robot_alignment', 'N/A')}")
        print(f"      Task: {r.get('has_task')} | Deictic: {r.get('has_deictic')} | Object: \"{r.get('mentioned_object', 'none')}\"")

        if decision["grounding"]:
            print(f"      Grounding [{decision['grounding_method']}]: {decision['grounding'][:80]}...")

        print(f"      → DECISION: {decision['addressee']}, act={decision['addressed']}")
        if decision["task"]:
            print(f"      → TASK: {decision['task'][:60]}...")
        print()
"""
Method C: Combined approach — Gemini for language + Method A for spatial reasoning.
Uses the VLM as a language understanding module, then feeds its output
into the structured spatial pipeline for grounding and navigation.
"""
import os
import json
import base64
import numpy as np
from PIL import Image
import io

from google import genai
from perception import load_sequence, get_context, cluster_utterances
from method_a import (compute_gaze_direction_world, compute_forward_direction_world,
                      extract_task_from_words, OBJECT_LOCATION_MAP)
from doa import estimate_doa, classify_speaker

_GEMINI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-2.5-flash"


def frame_to_base64(rgb_frame):
    """Convert numpy RGB frame to base64-encoded JPEG."""
    rgb_display = np.rot90(rgb_frame, k=3)
    img = Image.fromarray(rgb_display)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def vlm_understand_language(ctx, utterances, current_idx):
    """Use Gemini ONLY for language understanding — no spatial reasoning.

    Ask: what does the utterance mean? Is there a task? What object is referenced?
    Don't ask: where is the object? How far away? Which direction?
    """
    conversation_history = ""
    if utterances is not None and current_idx is not None:
        for i in range(max(0, current_idx - 3), current_idx + 1):
            utt = utterances[i]
            marker = ">>>" if i == current_idx else "   "
            conversation_history += f"{marker} [{i}] {utt['text']}\n"

    current_words = " ".join(ctx["speech_words"]["written"].tolist())

    prompt = f"""You are analyzing a kitchen conversation between two people. Focus ONLY on language understanding.

IMPORTANT DISTINCTIONS:
- "I'll get you some creamer" → HAS ACTION (physically fetching an object)
- "Let me get you a spoon" → HAS ACTION (physically fetching an object)
- "Imma put that back" → HAS ACTION (physically moving an object)
- "We got this, the popcorn for you" → NO ACTION (statement of possession, not movement)
- "Is it sealed?" → NO ACTION (question, no physical movement)
- "It's pretty good" → NO ACTION (opinion/comment)
- "Alright, is that it?" → NO ACTION (question about completion)
- "Oh yeah, it's fine" → NO ACTION (confirmation)
- "Here you go" → NO ACTION (handing over, already complete)

An action means someone WILL physically move, fetch, place, or retrieve an object. Questions, comments, confirmations, and statements about possession are NOT actions.

CONVERSATION:
{conversation_history}

CURRENT SPEECH (±3s): {current_words}

About the CURRENT utterance (marked with >>>):

Respond with ONLY a JSON object (no markdown, no backticks):
{{
    "has_action": true/false,
    "target_object": "specific object name" or null,
    "deictic_referent": "what 'that'/'this'/'it' refers to based on conversation" or null,
    "is_filler": true/false,
    "reasoning": "brief explanation"
}}"""


    frame_b64 = frame_to_base64(ctx["rgb_frame"])

    try:
        response = _GEMINI_CLIENT.models.generate_content(
            model=_MODEL,
            contents=[{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}}
                ]
            }]
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"    VLM language error: {e}")
        return {"has_action": False, "target_object": None,
                "deictic_referent": None, "is_filler": True,
                "reasoning": f"Error: {e}"}


def method_c_decide(ctx, handles, point_cloud_xyz, utterances, current_idx):
    """Combined approach: VLM language understanding + structured spatial pipeline.

    Step 1: Gemini analyzes the language (what object? is there a task?)
    Step 2: Method A's spatial pipeline grounds it (where is the object? navigation route?)
    """
    from object_finder import detect_objects, project_detection_to_world, find_object_world_position

    reasoning = {}

    # ========== STEP 1: VLM Language Understanding ==========
    vlm_result = vlm_understand_language(ctx, utterances, current_idx)
    reasoning["vlm_has_action"] = vlm_result.get("has_action", False)
    reasoning["vlm_target"] = vlm_result.get("target_object")
    reasoning["vlm_deictic"] = vlm_result.get("deictic_referent")
    reasoning["vlm_is_filler"] = vlm_result.get("is_filler", True)
    reasoning["vlm_reasoning"] = vlm_result.get("reasoning", "")

    has_task = vlm_result.get("has_action", False) and not vlm_result.get("is_filler", True)

    # Determine the target object name from VLM
    target_name = vlm_result.get("target_object")
    if target_name is None and vlm_result.get("deictic_referent"):
        target_name = vlm_result["deictic_referent"]

    # ========== STEP 2: Spatial Pipeline (from Method A) ==========

    # DoA
    if ctx["audio_window"] is not None and len(ctx["audio_window"]) > 1000:
        az_rad, spectrum, grid = estimate_doa(ctx["audio_window"])
        label, fwd_e, off_e, off_az_deg, peak_az_deg = classify_speaker(
            spectrum, grid, dominance_ratio=1.05
        )
        reasoning["doa_peak_deg"] = peak_az_deg
        reasoning["doa_speaker"] = label
    else:
        label = "unknown"
        reasoning["doa_speaker"] = "no_audio"

    # Gaze
    gaze_world = compute_gaze_direction_world(ctx)
    forward_world = compute_forward_direction_world(ctx)

    if gaze_world is not None and ctx["position_world"] is not None:
        assumed_robot_pos = ctx["position_world"] + 2.0 * forward_world
        wearer_to_robot = assumed_robot_pos - ctx["position_world"]
        wearer_to_robot_norm = wearer_to_robot / (np.linalg.norm(wearer_to_robot) + 1e-8)
        gaze_robot_cos = float(np.dot(gaze_world, wearer_to_robot_norm))
        reasoning["gaze_robot_alignment"] = gaze_robot_cos
    else:
        gaze_robot_cos = 0.0

    # ========== STEP 3: Spatial Grounding (using VLM's object name) ==========
    target_pos = None
    target_label = None
    grounding_method = None
    grounding_reasoning = None

    if has_task and target_name and handles is not None:
        wearer_pos = ctx["position_world"]
        frame = ctx["rgb_frame"]

        # Use GroundingDINO to find the object that Gemini named
        synonyms = {
            "creamer": "creamer. cream. cream bottle.",
            "cream": "cream. creamer. cream bottle.",
            "creamer carton": "creamer. cream. cream bottle. carton.",
            "spoon": "spoon. utensil.",
            "popcorn": "popcorn. popcorn bag. snack bag.",
        }
        query = synonyms.get(target_name.lower(), f"{target_name}.")

        detections = detect_objects(frame, query, confidence_threshold=0.2)

        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            best["timestamp_ns"] = ctx["query_ts_ns"]
            det_world = project_detection_to_world(best, handles, point_cloud_xyz)

            if det_world is not None:
                target_pos = det_world
                target_label = target_name
                grounding_method = "vlm+spatial"
                grounding_reasoning = (
                    f"VLM identified \"{target_name}\" as target. "
                    f"GroundingDINO found it in frame (conf: {best['confidence']:.2f}). "
                    f"Projected to ({det_world[0]:.1f}, {det_world[1]:.1f}, {det_world[2]:.1f})."
                )

        if target_pos is None:
            # Scan sequence
            world_pos, best_det = find_object_world_position(
                handles, point_cloud_xyz, query, num_frames=10
            )
            if world_pos is not None:
                target_pos = world_pos
                target_label = target_name
                grounding_method = "vlm+spatial"
                grounding_reasoning = (
                    f"VLM identified \"{target_name}\". "
                    f"Found via sequence scan at ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {world_pos[2]:.1f})."
                )

    reasoning["target_pos"] = target_pos.tolist() if target_pos is not None else None
    reasoning["target_label"] = target_label
    reasoning["grounding_method"] = grounding_method
    reasoning["grounding"] = grounding_reasoning

    # ========== Decision: COMBINE VLM language + spatial signals ==========
    # This is what makes Method C different from just "Method B + grounding"
    # 
    # VLM determines: is there a task? what object?
    # Spatial signals determine: should we trust the VLM's judgment?
    #   - DoA: is the wearer the one speaking? (not someone else)
    #   - Gaze: is the wearer engaged (looking forward, not away)?
    #   - Task confidence: does the VLM's reasoning make sense?

    vlm_says_task = has_task
    doa_says_wearer = (label in ["wearer", "ambiguous"])
    gaze_engaged = gaze_robot_cos > 0.5  # looser than Method A's 0.7

    # Combined decision logic:
    # Act only if VLM detects a task AND spatial signals support it
    if vlm_says_task and doa_says_wearer and gaze_engaged:
        should_act = True
        addressee = "robot"
    elif vlm_says_task and not doa_says_wearer:
        # VLM thinks there's a task but DoA says someone else is speaking
        should_act = False
        addressee = "other_human"
        reasoning["override"] = "VLM detected task but DoA indicates other speaker"
    elif not vlm_says_task:
        should_act = False
        if vlm_result.get("is_filler", True):
            addressee = "no_one"
        else:
            addressee = "other_human"
    else:
        should_act = False
        addressee = "other_human"

    # Build output
    decision = {
        "addressed": should_act,
        "addressee": addressee,
        "task": vlm_result.get("reasoning") if should_act else None,
        "has_deictic": vlm_result.get("deictic_referent") is not None,
        "mentioned_object": target_name,
        "target_pos": target_pos if should_act else None,
        "target_label": target_label if should_act else None,
        "grounding_method": grounding_method if should_act else None,
        "grounding": grounding_reasoning if should_act else None,
        "reasoning": reasoning,
    }

    return decision


if __name__ == "__main__":
    from visualization import load_filtered_points

    print("Loading sequence...")
    h = load_sequence()

    print("Filtering point cloud...")
    pc_xyz = load_filtered_points(h)
    print(f"  {len(pc_xyz)} points\n")

    utterances = cluster_utterances(h["speech"])
    print(f"  Found {len(utterances)} utterances\n")

    for i, utt in enumerate(utterances):
        print(f"  [{i}] {utt['text'][:60]}{'...' if len(utt['text']) > 60 else ''}")

    print(f"\n{'='*70}")
    print("Running Method C (VLM language + spatial grounding) on each utterance...\n")

    for i, utt in enumerate(utterances):
        ctx = get_context(h, utt["mid_ns"])
        if ctx["pose"] is None:
            print(f"  [{i}] OUT OF RANGE\n")
            continue

        decision = method_c_decide(ctx, h, pc_xyz, utterances, i)

        r = decision["reasoning"]
        print(f"  [{i}] t={utt['mid_ns']/1e9:.2f}s — \"{utt['text'][:50]}{'...' if len(utt['text']) > 50 else ''}\"")
        print(f"      VLM: action={r.get('vlm_has_action')} | target=\"{r.get('vlm_target', 'none')}\" | deictic=\"{r.get('vlm_deictic', 'none')}\"")
        print(f"      VLM reasoning: {r.get('vlm_reasoning', '?')[:70]}...")

        if decision["grounding"]:
            print(f"      Grounding [{decision['grounding_method']}]: {decision['grounding'][:70]}...")

        print(f"      → DECISION: {decision['addressee']}, act={decision['addressed']}")
        print()
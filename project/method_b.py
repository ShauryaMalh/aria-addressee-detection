"""
Method B: VLM baseline for addressee detection + task extraction.
Sends the RGB frame + transcript to Gemini and lets it reason holistically
without explicit spatial computation (no DoA, no gaze projection, no point cloud).
"""
import os
import json
import base64
import numpy as np
from PIL import Image
import io

from google import genai
from perception import load_sequence, get_context, cluster_utterances


# Initialize Gemini client
_GEMINI_CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_MODEL = "gemini-2.5-flash"


def frame_to_base64(rgb_frame):
    """Convert numpy RGB frame to base64-encoded JPEG for the API."""
    # Rotate to correct Aria's camera mounting (same as visualization)
    rgb_display = np.rot90(rgb_frame, k=3)
    img = Image.fromarray(rgb_display)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def method_b_decide(ctx, utterances=None, current_utterance_idx=None):
    """Run Method B: send frame + transcript to Gemini for holistic reasoning.

    Args:
        ctx: output of get_context() for this timestamp.
        utterances: list of all utterances (for conversation context).
        current_utterance_idx: index of the current utterance in the list.

    Returns:
        decision: dict matching Method A's output format for comparison.
    """
    # Build conversation context from previous utterances
    conversation_history = ""
    if utterances is not None and current_utterance_idx is not None:
        for i in range(max(0, current_utterance_idx - 3), current_utterance_idx + 1):
            utt = utterances[i]
            marker = ">>>" if i == current_utterance_idx else "   "
            conversation_history += f"{marker} [{i}] {utt['text']}\n"
    
    # Current speech window
    current_words = " ".join(ctx["speech_words"]["written"].tolist())

    # Build the prompt
    prompt = f"""You are a spatial reasoning agent embedded in AR glasses worn by Person A (the wearer). Person B is also present. Your job is to ASSIST Person A by understanding the conversation and identifying actionable moments where you can help.

IMPORTANT: When Person A announces an action they intend to perform (e.g., "I'll get you some creamer", "let me get you a spoon", "Imma put that back"), this IS an actionable task — you should identify the target object and help plan the action, even though Person A is the one who will execute it. You are an assistant helping the wearer, not waiting for direct commands.

The key questions are:
1. Does the current utterance involve a PHYSICAL ACTION (getting, putting, moving, fetching an object)?
2. If yes, what is the TARGET OBJECT?
3. If the utterance uses "that", "this", "it" — what does it refer to based on conversation context?

Non-actionable utterances are: casual comments ("pretty good"), questions without action ("is it sealed?"), greetings, confirmations ("okay", "alright").

CONVERSATION CONTEXT (>>> marks the current utterance):
{conversation_history}

CURRENT SPEECH WINDOW (±3 seconds): {current_words}

The image shows what Person A is currently seeing through their AR glasses.

Respond with ONLY a JSON object (no markdown, no backticks) with these fields:
{{
    "addressed": true/false (is there an actionable task the agent should help with?),
    "addressee": "robot" if there is a task, "other_human" if just conversation, "no_one" if filler,
    "has_task": true/false,
    "task_description": "brief description" or null,
    "has_deictic": true/false,
    "target_object": "name of the object" or null,
    "target_object_reasoning": "explain what the target is and why" or null,
    "deictic_resolution": "if 'that'/'this' was used, what does it refer to and why?" or null,
    "confidence": "high" or "medium" or "low",
    "reasoning": "1-2 sentence explanation",
    "spatial_estimate": "estimate the direction and distance from the wearer to the target object (e.g., '2 meters to the left')" or null,
    "can_plot_route": true/false (could you plot an exact navigation route with coordinates?)
}}"""

    # Encode the frame
    frame_b64 = frame_to_base64(ctx["rgb_frame"])

    # Call Gemini with image + text
    try:
        response = _GEMINI_CLIENT.models.generate_content(
            model=_MODEL,
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": frame_b64
                            }
                        }
                    ]
                }
            ]
        )

        # Parse the JSON response
        response_text = response.text.strip()
        # Remove markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            response_text = response_text.rsplit("```", 1)[0]
        
        result = json.loads(response_text)

    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        print(f"    Raw response: {response.text[:200]}")
        result = {
            "addressed": False,
            "addressee": "unknown",
            "has_task": False,
            "task_description": None,
            "has_deictic": False,
            "target_object": None,
            "target_object_reasoning": None,
            "confidence": "low",
            "reasoning": f"Failed to parse VLM response: {str(e)}"
        }
    except Exception as e:
        print(f"    API error: {e}")
        result = {
            "addressed": False,
            "addressee": "error",
            "has_task": False,
            "task_description": None,
            "has_deictic": False,
            "target_object": None,
            "target_object_reasoning": None,
            "confidence": "low",
            "reasoning": f"API call failed: {str(e)}"
        }

    # Convert to Method A's output format for easy comparison
    decision = {
        "addressed": result.get("addressed", False),
        "addressee": result.get("addressee", "unknown"),
        "task": result.get("task_description"),
        "has_deictic": result.get("has_deictic", False),
        "mentioned_object": result.get("target_object"),
        "target_pos": None,
        "target_label": result.get("target_object"),
        "grounding_method": "vlm" if result.get("target_object") else None,
        "grounding": result.get("target_object_reasoning"),
        "spatial_estimate": result.get("spatial_estimate"),
        "can_plot_route": result.get("can_plot_route", False),
        "reasoning": {
            "vlm_addressee": result.get("addressee"),
            "vlm_confidence": result.get("confidence"),
            "vlm_reasoning": result.get("reasoning"),
            "vlm_has_task": result.get("has_task"),
            "vlm_has_deictic": result.get("has_deictic"),
            "vlm_target": result.get("target_object"),
            "vlm_target_reasoning": result.get("target_object_reasoning"),
            "vlm_spatial_estimate": result.get("spatial_estimate"),
            "vlm_can_plot_route": result.get("can_plot_route"),
        },
    }

    return decision


if __name__ == "__main__":
    print("Loading sequence...")
    h = load_sequence()

    print("Clustering utterances...")
    utterances = cluster_utterances(h["speech"])
    print(f"  Found {len(utterances)} utterances\n")

    for i, utt in enumerate(utterances):
        print(f"  [{i}] {utt['text'][:60]}{'...' if len(utt['text']) > 60 else ''}")

    print(f"\n{'='*70}")
    print("Running Method B (Gemini VLM) on each utterance...\n")

    for i, utt in enumerate(utterances):
        ctx = get_context(h, utt["mid_ns"])
        if ctx["pose"] is None:
            print(f"  [{i}] OUT OF RANGE\n")
            continue

        decision = method_b_decide(ctx, utterances=utterances, current_utterance_idx=i)

        r = decision["reasoning"]
        print(f"  [{i}] t={utt['mid_ns']/1e9:.2f}s — \"{utt['text'][:50]}{'...' if len(utt['text']) > 50 else ''}\"")
        print(f"      Addressee: {r.get('vlm_addressee', '?')} | Confidence: {r.get('vlm_confidence', '?')}")
        print(f"      Task: {r.get('vlm_has_task', '?')} | Target: {r.get('vlm_target', 'none')}")
        print(f"      Reasoning: {r.get('vlm_reasoning', '?')[:80]}...")
        print(f"      → DECISION: {decision['addressee']}, act={decision['addressed']}")
        if decision.get("spatial_estimate"):
            print(f"      Spatial estimate: {decision['spatial_estimate']}")
        print(f"      Can plot route: {decision.get('can_plot_route', False)}")
        print() 
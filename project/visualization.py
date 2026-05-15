"""
Top-down map visualization for the addressee detection demo.
Renders wearer position, other person estimate, target objects,
and navigation routes on a bird's-eye view of the point cloud.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for attu (no display)
import matplotlib.pyplot as plt
from pathlib import Path

from projectaria_tools.core.mps.utils import filter_points_from_confidence


def load_filtered_points(handles):
    """Filter the point cloud for high-confidence 3D points."""
    filtered = filter_points_from_confidence(
        handles["points"],
        threshold_invdep=0.001,
        threshold_dep=0.15
    )
    # Extract xyz positions as (N, 3) array
    positions = np.array([p.position_world for p in filtered])
    return positions

def compute_other_person_world(ctx, doa_az_rad, estimated_distance=2.0):
    """Estimate the other person's world position from DoA bearing."""
    if ctx["pose"] is None:
        return None

    direction_device = np.array([
        np.cos(doa_az_rad),
        0.0,
        np.sin(doa_az_rad),
    ])

    R = ctx["pose"].rotation().to_matrix()
    direction_world = R @ direction_device
    direction_world[2] = 0
    norm = np.linalg.norm(direction_world)
    if norm > 0:
        direction_world /= norm

    other_pos = ctx["position_world"] + estimated_distance * direction_world
    return other_pos


def render_map(point_cloud_xyz, ctx, doa_az_rad, decision, handles,
               target_object_pos=None, target_label="Target",
               other_person_pos=None, case_label="", utterance="",
               save_path="map.png", zoom_radius=3.5):
    """Render a two-panel figure: top-down map + RGB frame with info."""

    fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(18, 9),
                                           gridspec_kw={'width_ratios': [1.2, 1]})

    wearer_pos = ctx["position_world"]

    # ==================== LEFT PANEL: TOP-DOWN MAP ====================

    # Crop point cloud
    pc_xy = point_cloud_xyz[:, :2]
    mask = (
        (np.abs(pc_xy[:, 0] - wearer_pos[0]) < zoom_radius) &
        (np.abs(pc_xy[:, 1] - wearer_pos[1]) < zoom_radius)
    )
    local_pc = pc_xy[mask]

    # Room geometry
    ax_map.scatter(local_pc[:, 0], local_pc[:, 1],
                   s=0.8, c='#999999', alpha=0.6, zorder=1, rasterized=True)

    # --- Wearer trajectory trace (last 10 seconds) ---
    """
    if handles is not None:
        from projectaria_tools.core.mps.utils import get_nearest_pose
        trace_points = []
        for dt in np.arange(-10.0, 0.5, 0.5):  # every 0.5s for last 10s
            trace_ts = ctx["query_ts_ns"] + int(dt * 1e9)
            trace_pose = get_nearest_pose(handles["trajectory"], trace_ts)
            if trace_pose is not None:
                pos = trace_pose.transform_world_device.translation().flatten()
                trace_points.append(pos[:2])
        if len(trace_points) > 1:
            trace_arr = np.array(trace_points)
            ax_map.plot(trace_arr[:, 0], trace_arr[:, 1],
                        '-', color='#BBDEFB', lw=2.5, alpha=0.7, zorder=2,
                        label='Recent path')
    """
    # Wearer
    ax_map.scatter(wearer_pos[0], wearer_pos[1], s=300, c='#2196F3',
                   marker='o', zorder=10, edgecolors='white', linewidth=2)
    ax_map.annotate('Wearer (A)', (wearer_pos[0], wearer_pos[1]),
                    textcoords="offset points", xytext=(12, -15),
                    fontsize=11, fontweight='bold', color='#1565C0',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Forward direction
    from method_a import compute_forward_direction_world, compute_gaze_direction_world
    forward = compute_forward_direction_world(ctx)
    '''
    # Forward direction (removed — too similar to gaze, adds clutter) 
    if forward is not None:
        fwd_end = wearer_pos[:2] + 1.0 * forward[:2]
        ax_map.annotate('', xy=fwd_end, xytext=wearer_pos[:2],
                        arrowprops=dict(arrowstyle='->', color='#90CAF9',
                                        lw=2, mutation_scale=15), zorder=6)
    '''

    # Gaze direction
    gaze_world = compute_gaze_direction_world(ctx)
    if gaze_world is not None:
        gaze_end = wearer_pos[:2] + 1.5 * gaze_world[:2]
        ax_map.annotate('', xy=gaze_end, xytext=wearer_pos[:2],
                        arrowprops=dict(arrowstyle='->', color='#1565C0',
                                        lw=3, mutation_scale=20), zorder=7)
        ax_map.annotate('gaze', gaze_end, textcoords="offset points",
                        xytext=(5, 5), fontsize=9, fontweight='bold',
                        color='#1565C0')

    # --- DoA bearing (removed — always dominated by wearer's voice) ---
    '''if doa_az_rad is not None and ctx["pose"] is not None:
        R = ctx["pose"].rotation().to_matrix()
        doa_device = np.array([np.cos(doa_az_rad), 0.0, np.sin(doa_az_rad)])
        doa_world = R @ doa_device
        doa_end = wearer_pos[:2] + 2.0 * doa_world[:2]
        ax_map.plot([wearer_pos[0], doa_end[0]], [wearer_pos[1], doa_end[1]],
                    '--', color='#FF9800', lw=2, alpha=0.7, zorder=5)
        ax_map.annotate(f'DoA', doa_end, textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color='#FF9800')
    '''
    # Other person
    if other_person_pos is not None:
        ax_map.scatter(other_person_pos[0], other_person_pos[1], s=300, c='#F44336',
                       marker='o', zorder=10, edgecolors='white', linewidth=2)
        ax_map.annotate('Other Person (B)', other_person_pos[:2],
                        textcoords="offset points", xytext=(15, -15),
                        fontsize=10, fontweight='bold', color='#D32F2F',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    # Case-specific elements
    addressee = decision.get("addressee", "")
    should_act = decision.get("addressed", False)

    if should_act and target_object_pos is not None:
        # Case 1: target object + navigation route
        ax_map.scatter(target_object_pos[0], target_object_pos[1], s=300, c='#4CAF50',
                       marker='s', zorder=10, edgecolors='white', linewidth=2)
        ax_map.annotate(target_label, target_object_pos[:2],
                        textcoords="offset points", xytext=(-15, -20),
                        fontsize=10, fontweight='bold', color='#2E7D32',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

        # Navigation route arrows only (no step labels)
        route_points = [wearer_pos[:2], target_object_pos[:2]]
        if other_person_pos is not None:
            route_points.append(other_person_pos[:2])

        for i in range(len(route_points) - 1):
            ax_map.annotate('', xy=route_points[i+1], xytext=route_points[i],
                            arrowprops=dict(arrowstyle='->', color='#333333',
                                            lw=2.5, mutation_scale=18,
                                            linestyle='dashed'), zorder=4)
            # Step label
            mid = (route_points[i] + route_points[i+1]) / 2
            step_labels = ["Step 1: Go to target", "Step 2: Return to person"]
            ax_map.annotate(step_labels[i] if i < len(step_labels) else "",
                            mid, textcoords="offset points", xytext=(0, -12),
                            fontsize=8, color='#555555', ha='center',
                            fontstyle='italic')

    elif not should_act:
        # Case 2: orient arrow — point toward the other person if detected
        if other_person_pos is not None:
            # Direction from wearer to other person
            to_person = other_person_pos[:2] - wearer_pos[:2]
            dist_to_person = np.linalg.norm(to_person)
            if dist_to_person > 0.1:
                to_person_norm = to_person / dist_to_person
                orient_end = wearer_pos[:2] + 1.8 * to_person_norm
                ax_map.annotate('', xy=orient_end, xytext=wearer_pos[:2],
                                arrowprops=dict(arrowstyle='->', color='#FF6F00',
                                                lw=4, mutation_scale=25), zorder=8)
                ax_map.annotate('Orient here', orient_end, textcoords="offset points",
                                xytext=(8, 8), fontsize=11, fontweight='bold',
                                color='#FF6F00')
        elif doa_az_rad is not None and ctx["pose"] is not None:
            # Fallback to DoA if no person detected
            R = ctx["pose"].rotation().to_matrix()
            doa_device = np.array([np.cos(doa_az_rad), 0.0, np.sin(doa_az_rad)])
            doa_world = R @ doa_device
            orient_end = wearer_pos[:2] + 1.8 * doa_world[:2]
            ax_map.annotate('', xy=orient_end, xytext=wearer_pos[:2],
                            arrowprops=dict(arrowstyle='->', color='#FF6F00',
                                            lw=4, mutation_scale=25), zorder=8)
            ax_map.annotate('Orient here', orient_end, textcoords="offset points",
                            xytext=(8, 8), fontsize=11, fontweight='bold',
                            color='#FF6F00')

    # Map formatting
    ax_map.set_xlim(wearer_pos[0] - zoom_radius, wearer_pos[0] + zoom_radius)
    ax_map.set_ylim(wearer_pos[1] - zoom_radius, wearer_pos[1] + zoom_radius)
    ax_map.set_xlabel('X (meters)', fontsize=11)
    ax_map.set_ylabel('Y (meters)', fontsize=11)
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.15)
    ax_map.set_title('Spatial Map (Top-Down)', fontsize=12, fontweight='bold')

    # Scale bar
    bar_x = wearer_pos[0] - zoom_radius + 0.3
    bar_y = wearer_pos[1] - zoom_radius + 0.3
    ax_map.plot([bar_x, bar_x + 1.0], [bar_y, bar_y], 'k-', lw=3)
    ax_map.text(bar_x + 0.5, bar_y + 0.15, '1m', ha='center', fontsize=9)

    # Legend
    ax_map.legend(loc='lower right', fontsize=8, framealpha=0.9)

    # ==================== RIGHT PANEL: RGB FRAME + INFO ====================

    # Show the RGB frame
# Aria RGB camera is mounted rotated — correct for display
    rgb_display = np.rot90(ctx["rgb_frame"], k=3)  # 90° counterclockwise
    ax_info.imshow(rgb_display)
    ax_info.set_title('Egocentric View (RGB)', fontsize=12, fontweight='bold')
    ax_info.axis('off')

    # Build info text below the image
    r = decision.get("reasoning", {})
    info_parts = []
    info_parts.append(f"Utterance: \"{utterance[:70]}{'...' if len(utterance) > 70 else ''}\"")
    info_parts.append("")

    # Transcript window
    words = " ".join(ctx["speech_words"]["written"].tolist())
    if len(words) > 80:
        words = words[:80] + "..."
    info_parts.append(f"Transcript (±3s): {words}")
    info_parts.append("")

    # Decision
    info_parts.append(f"{'─'*45}")
    if should_act:
        info_parts.append(f"DECISION: Act on behalf of wearer")
        if decision.get("target_label"):
            info_parts.append(f"Target: {decision['target_label']}")
            method = decision.get("grounding_method", "?")
            if method == "direct":
                conf = r.get("grounding", "")
                # Extract confidence from grounding string
                info_parts.append(f"Found via: direct visual search (GroundingDINO)")
            elif method == "deictic":
                info_parts.append(f"Found via: deictic resolution (gaze + vision)")
            elif method == "gaze":
                info_parts.append(f"Found via: gaze ray intersection")
            elif method == "lookup":
                info_parts.append(f"Found via: commonsense lookup + visual confirm")
        else:
            info_parts.append(f"Target: could not be grounded")
    else:
        info_parts.append(f"DECISION: No action — orient toward conversation")
    info_parts.append(f"{'─'*45}")
    info_parts.append("")

    # Reasoning signals
    info_parts.append("Signals used:")
    if r.get("doa_peak_deg") is not None:
        speaker = r.get('doa_speaker', '?')
        info_parts.append(f"  Audio (DoA): peak at {r['doa_peak_deg']:+.0f}° → {speaker}")
    if r.get("gaze_robot_alignment") is not None:
        info_parts.append(f"  Spatial (gaze): {r['gaze_robot_alignment']:.2f} alignment to forward")
    info_parts.append(f"  Language: task={'yes' if r.get('has_task') else 'no'}, "
                      f"deictic={'yes' if r.get('has_deictic') else 'no'}")
    if r.get("mentioned_object"):
        info_parts.append(f"  Object mentioned: \"{r['mentioned_object']}\"")
    
    # Grounding detail
    if r.get("grounding"):
        info_parts.append("")
        grounding_text = r["grounding"]
        # Wrap long grounding text
        if len(grounding_text) > 90:
            info_parts.append(f"  {grounding_text[:90]}")
            info_parts.append(f"  {grounding_text[90:160]}")
        else:
            info_parts.append(f"  {grounding_text}")

    info_text = "\n".join(info_parts)
    fig.text(0.55, 0.02, info_text, fontsize=8.5, fontfamily='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                       edgecolor='#CCCCCC'))

    # Main title
    fig.suptitle(case_label, fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    from perception import load_sequence, get_context, cluster_utterances
    from doa import estimate_doa
    from method_a import method_a_decide

    print("Loading sequence...")
    h = load_sequence()

    print("Filtering point cloud...")
    pc_xyz = load_filtered_points(h)
    print(f"  {len(pc_xyz)} points after filtering")

    # Find other person using GroundingDINO
    print("\n--- Finding other person ---")
    from object_finder import detect_objects, scan_for_objects, project_detection_to_world
    person_detections = scan_for_objects(h, "person. human.", num_frames=20)
    person_world_pos = None
    if person_detections:
        best_person = max(person_detections, key=lambda d: d["confidence"])
        best_person["timestamp_ns"] = best_person.get("timestamp_ns", 0)
        person_world_pos = project_detection_to_world(best_person, h, pc_xyz)
        if person_world_pos is not None:
            print(f"  Person at: ({person_world_pos[0]:.2f}, {person_world_pos[1]:.2f}, {person_world_pos[2]:.2f})")

    # Auto-detect all utterances
    print("\n--- Clustering utterances ---")
    utterances = cluster_utterances(h["speech"])
    print(f"  Found {len(utterances)} utterances\n")

    # Run Method A on each and render maps
    case_1_count = 0
    case_2_count = 0

    for i, utt in enumerate(utterances):
        ctx = get_context(h, utt["mid_ns"])
        if ctx["pose"] is None:
            continue

        az_rad, _, _ = estimate_doa(ctx["audio_window"])
        decision = method_a_decide(ctx, handles=h, point_cloud_xyz=pc_xyz, utterances=utterances)

        should_act = decision["addressed"]

        # Use detected person, fall back to DoA estimate
        other_pos = person_world_pos if person_world_pos is not None else \
                    compute_other_person_world(ctx, az_rad, estimated_distance=2.0)

        if should_act:
            case_1_count += 1
            # Get target from decision (already computed by method_a_decide)
            target_pos = decision.get("target_pos")
            target_label = decision.get("target_label", "Target")
            grounding = decision.get("grounding_method", "unknown")

            if target_label and grounding:
                display_label = f"{target_label} [{grounding}]"
            else:
                display_label = "Unknown target [failed]"

            save_path = f"/homes/iws/shaurm/projectaria_sandbox/project/case1_utt{i}.png"
            print(f"  Rendering Case 1 for [{i}]: \"{utt['text'][:50]}...\"")

            render_map(
                pc_xyz, ctx, az_rad, decision, handles=h,
                target_object_pos=target_pos,
                target_label=display_label,
                other_person_pos=other_pos,
                case_label=f"Case 1: Task-Directed Speech (utterance {i})",
                utterance=utt["text"][:80],
                save_path=save_path,
                zoom_radius=3.5
            )
        else:
            case_2_count += 1
            save_path = f"/homes/iws/shaurm/projectaria_sandbox/project/case2_utt{i}.png"
            print(f"  Rendering Case 2 for [{i}]: \"{utt['text'][:50]}...\"")

            render_map(
                pc_xyz, ctx, az_rad, decision, handles=h,
                other_person_pos=other_pos,
                case_label=f"Case 2: Non-Addressed Speech (utterance {i})",
                utterance=utt["text"][:80],
                save_path=save_path,
                zoom_radius=3.5
            )

    print(f"\nDone. Rendered {case_1_count} Case 1 maps and {case_2_count} Case 2 maps.")
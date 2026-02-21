"""
visual_generator.py
====================
Calls visualization.py from audio_module using plot_prediction_summary().

IMPORTANT: visualization.py uses relative imports internally, so it must be
imported as audio_module.visualization (package import), not standalone.
"""

import os
import sys

# ── Resolve paths ─────────────────────────────────────────────────────────────
_UTILS_DIR     = os.path.dirname(os.path.abspath(__file__))   # amberisai/utils/
_AMBERISAI_DIR = os.path.dirname(_UTILS_DIR)                  # amberisai/
_HACKATHON_DIR = os.path.dirname(_AMBERISAI_DIR)              # kal ki hackathon/

AUDIO_MODULE_DIR = os.path.join(_HACKATHON_DIR, 'audio_module')
VISUALS_DIR      = os.path.join(_AMBERISAI_DIR, 'static', 'visuals')

# Add HACKATHON_DIR so audio_module is importable as a package
if _HACKATHON_DIR not in sys.path:
    sys.path.insert(0, _HACKATHON_DIR)


def generate_visual(
    audio_file_path: str,
    session_id: str,
    primary_condition: str,
    analysis: dict
) -> str | None:
    """
    Generate a PNG visualization using visualization.py's plot_prediction_summary().

    Draws:
      - Left panel : raw waveform of the cry audio
      - Right panel: probability bar chart for all conditions

    Returns URL string like "/static/visuals/sess_xxx_hungry.png" or None.
    """
    os.makedirs(VISUALS_DIR, exist_ok=True)

    output_filename = f"{session_id}_{primary_condition}.png"
    output_path     = os.path.join(VISUALS_DIR, output_filename)

    try:
        import matplotlib.pyplot as plt

        # Import as package so relative imports inside visualization.py work
        from audio_module.utils import load_audio_with_checks
        from audio_module.visualization import plot_prediction_summary

        # Load audio for the waveform panel
        y, sr, _ = load_audio_with_checks(audio_file_path)

        # Generate and save the figure
        fig, axes = plot_prediction_summary(
            prediction_result=analysis,
            y=y,
            sr=sr,
            save_path=output_path
        )
        plt.close(fig)

        if os.path.exists(output_path):
            print(f"[Visual] PNG saved: {output_path}")
            return f"/static/visuals/{output_filename}"
        else:
            print("[Visual] plot_prediction_summary ran but PNG not found — using fallback")
            return _fallback_chart(output_path, primary_condition, analysis)

    except Exception as e:
        print(f"[Visual] visualization.py failed: {e} — using fallback chart")
        return _fallback_chart(output_path, primary_condition, analysis)


def _fallback_chart(output_path: str, primary_condition: str, analysis: dict) -> str | None:
    """
    Clean dark-themed fallback chart if visualization.py fails for any reason.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        BG     = '#1a1a2e'
        PANEL  = '#16213e'
        ACCENT = '#e94560'
        DIM    = '#0f3460'
        WHITE  = 'white'
        GREY   = '#888888'

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)

        all_probs = analysis.get('all_probabilities', {})
        if all_probs:
            sorted_items = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            labels  = [k for k, _ in sorted_items]
            values  = [v for _, v in sorted_items]
            colours = [ACCENT if l == primary_condition else DIM for l in labels]

            bars = ax.barh(labels, values, color=colours, edgecolor='none', height=0.55)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1%}', va='center', ha='left',
                    color=WHITE, fontsize=9
                )
            ax.set_xlim(0, 1.18)
        else:
            ax.text(0.5, 0.5, 'No probability data', ha='center',
                    va='center', color=GREY, transform=ax.transAxes)

        confidence = analysis.get('confidence', 0)
        ax.set_title(
            f'AmberisAI  •  {primary_condition.upper()}  •  {confidence:.1%} confidence',
            color=WHITE, fontsize=13, fontweight='bold', pad=12
        )
        ax.set_xlabel('Confidence', color=WHITE)
        ax.tick_params(colors=WHITE)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=BG, edgecolor='none')
        plt.close(fig)

        print(f"[Visual] Fallback PNG saved: {output_path}")
        return f"/static/visuals/{os.path.basename(output_path)}"

    except Exception as e2:
        print(f"[Visual] Fallback also failed: {e2}")
        return None
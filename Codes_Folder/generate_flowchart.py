import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_methodology_flowchart(output_path: str):
    fig, ax = plt.subplots(figsize=(14, 18), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Palette
    c_blue = '#1E64B4'
    c_blue_bg = '#EBF3FA'
    c_green = '#2E7D32'
    c_green_bg = '#E8F5E9'
    c_orange = '#D84315'
    c_orange_bg = '#FBE9E7'
    c_purple = '#6A1B9A'
    c_purple_bg = '#F3E5F5'
    c_slate = '#37474F'
    c_slate_bg = '#ECEFF1'
    
    # Helper to draw a box
    def draw_box(x, y, w, h, text, title='', bg='#EBF3FA', ec='#1E64B4', title_color='#0D47A1', fontsize=10):
        # Shadow
        shadow = patches.FancyBboxPatch((x+0.6, y-0.6), w, h, boxstyle='round,pad=0.5,rounding_size=1.5',
                                        facecolor='#D0D7DE', edgecolor='none', alpha=0.5, zorder=1)
        ax.add_patch(shadow)
        # Main box
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5,rounding_size=1.5',
                                     facecolor=bg, edgecolor=ec, linewidth=1.8, zorder=2)
        ax.add_patch(box)
        
        # Text rendering with native matplotlib bold styling
        if title and text:
            ax.text(x + w/2, y + h - 1.1, title, ha='center', va='top', fontsize=fontsize+0.8,
                    fontweight='bold', color=title_color, zorder=3)
            ax.text(x + w/2, y + (h - 2.0)/2, text, ha='center', va='center', fontsize=fontsize-0.5,
                    color='#1A1A1A', zorder=3, linespacing=1.25)
        elif title and not text:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=fontsize+1,
                    fontweight='bold', color=title_color, zorder=3)
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
                    color='#1A1A1A', zorder=3, linespacing=1.25)

    # Helper to draw an arrow
    def draw_arrow(x1, y1, x2, y2, label='', color='#37474F', lw=2.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=18),
                    zorder=4)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx + 1.5, my, label, fontsize=9, color=color, fontweight='bold',
                    ha='left', va='center', bbox=dict(boxstyle='square,pad=0.2', fc='white', ec='none', alpha=0.85), zorder=5)

    # 1. Title Box
    draw_box(10, 93, 80, 5, "Predictive Maintenance in Metro Trains: End-to-End Dual-Evaluation Pipeline",
             title="METHODOLOGY ARCHITECTURE FLOWCHART", bg='#0D47A1', ec='#0D47A1', title_color='#FFFFFF', fontsize=11)
    ax.texts[-1].set_color('#EBF3FA')

    # 2. Raw Telemetry
    draw_box(15, 83.5, 70, 6.5, 
             "• MetroPT-3 (UCI): 10,773,588 rows (16 APU pressure/current/temp features)\n• MetroPT-2 (Zenodo): 7,116,940 rows (Identical sensor layout & bus telemetry)",
             title="Step 1: Raw Telemetry Streaming & Ingestion", bg=c_blue_bg, ec=c_blue)
    draw_arrow(50, 83.5, 50, 77.5)

    # 3. Resampling
    draw_box(15, 71, 70, 6.5, 
             "• Regular 10-second (0.1 Hz) mean-aggregation for thermodynamic continuous sensors\n• Last-state persistence for safety switches (LPS, COMP, DV_eletric) | 90% size reduction",
             title="Step 2: 10-Second Physical Resampling", bg=c_blue_bg, ec=c_blue)
    draw_arrow(50, 71, 50, 65)

    # 4. Failure Horizon Labeling
    draw_box(15, 58.5, 70, 6.5, 
             "• Ground-Truth logs from official maintenance records (Nature Sci. Data 2022; IJDS 2026)\n• 2-Hour Pre-Failure Lead Window ($[T_{fail} - 2h, T_{fail}]$) assigned prior to breakdown",
             title="Step 3: Defensible Ground-Truth Horizon Labeling", bg=c_orange_bg, ec=c_orange)
    draw_arrow(50, 58.5, 50, 52.5)

    # 5. Chronological Partitioning
    draw_box(15, 46, 70, 6.5, 
             "• Zero Temporal Leakage: MetroPT-3 (70/15/15) | MetroPT-2 (65/15/20)\n• Train split: Air leaks only | Val: Clean baseline | Test: Unseen oil leak (MetroPT-3)",
             title="Step 4: Strict Chronological Splitting & Leak-Free Scaling", bg=c_slate_bg, ec=c_slate)

    # Fork arrows to Balancing and Sliding Windows
    draw_arrow(32, 46, 32, 40)
    draw_arrow(68, 46, 68, 40)

    # 6a. Training Balancing
    draw_box(10, 33, 40, 7, 
             "• Exact 1:1 downsampling of normal windows\n• MetroPT-3 Train: 517 Normal + 517 Warning\n• MetroPT-2 Train: 365 Normal + 365 Warning",
             title="Step 5a: Training Set 1:1 Balancing", bg=c_green_bg, ec=c_green, fontsize=9.5)

    # 6b. Validation & Test Preservation
    draw_box(52, 33, 40, 7, 
             "• 100% Real-World Class Imbalance Preserved\n• MetroPT-3 Test: 23,547 Normal + 3,378 Fail\n• MetroPT-2 Test: 19,947 Normal + 3,767 Fail",
             title="Step 5b: Natural Test Set Evaluation", bg=c_orange_bg, ec=c_orange, fontsize=9.5)

    draw_arrow(30, 33, 45, 27)
    draw_arrow(72, 33, 55, 27)

    # 7. Model Architectures
    draw_box(15, 20.5, 70, 6.5, 
             "• Sliding Window Dataset: W = 60 steps (10 min), Stride S = 6 steps (1 min)\n• Candidate Models: LSTM (2-layer), GRU (2-layer), TCN (dilated causal residual convs)",
             title="Step 6: Sequence Windowing & Model Training Matrix", bg=c_purple_bg, ec=c_purple)

    # Fork arrows to Dual Evaluation
    draw_arrow(30, 20.5, 25, 14.5)
    draw_arrow(50, 20.5, 50, 14.5)
    draw_arrow(70, 20.5, 75, 14.5)

    # 8a. Task 1
    draw_box(4, 5.5, 29, 9, 
             "• Evaluates across 2h lead window AND full active outage\n• Positive Class = 3,378 / 3,767\n• High scores reflect ongoing outage detection (PR-AUC 0.97)",
             title="Task 1: Active Fault State Monitoring", bg=c_blue_bg, ec=c_blue, fontsize=8.5)

    # 8b. Validation Threshold Calibration
    draw_box(35.5, 5.5, 29, 9, 
             "• Unsupervised quantiles on healthy validation data (q90, q95, q98)\n• Resolves recall collapse at tau=0.5\n• Recovers 35.0% recall on oil leak under Task 1 monitoring",
             title="Validation Threshold Calibration", bg=c_slate_bg, ec=c_slate, fontsize=8.5)

    # 8c. Task 2
    draw_box(67, 5.5, 29, 9, 
             "• Pure 2h pre-failure warning\n• Active outage EXCLUDED\n• Positive Class = 120 windows\n• Base-rate precision ~49.6% at 0.61% FPR (PR-AUC 0.43)",
             title="Task 2: Anticipatory Pre-Failure Warning", bg=c_green_bg, ec=c_green, fontsize=8.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Flowchart successfully generated at: {output_path}")

if __name__ == '__main__':
    generate_methodology_flowchart('reports/figures/methodology_flowchart.png')

"""
Generate Clean Architecture Diagram for Trading Bot (without emojis)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")

# Colors
COLORS = {
    "main": "#00d4ff",
    "core": "#4CAF50",
    "support": "#FF9800",
    "notification": "#E91E63",
    "config": "#9C27B0",
    "data": "#607D8B",
    "external": "#795548",
    "text": "#ffffff",
    "subtext": "#aaaaaa",
    "box_bg": "#16213e",
    "border": "#0f3460",
}


def draw_box(ax, x, y, w, h, label, sublabel="", color=COLORS["core"], fontsize=10):
    """Draw a styled box with label"""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.5",
        facecolor=COLORS["box_bg"],
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(box)

    # Main label
    ax.text(
        x + w / 2,
        y + h / 2 + (0.8 if sublabel else 0),
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )

    # Sub label
    if sublabel:
        ax.text(
            x + w / 2,
            y + h / 2 - 1.2,
            sublabel,
            ha="center",
            va="center",
            fontsize=fontsize - 2,
            color=COLORS["subtext"],
            style="italic",
        )


def draw_section(ax, x, y, w, h, title, color):
    """Draw a section box with title"""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=1",
        facecolor="none",
        edgecolor=color,
        linewidth=1.5,
        linestyle="--",
        alpha=0.5,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h + 1.5,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=color,
    )


def draw_arrow(ax, start, end, color=COLORS["text"]):
    """Draw a styled arrow"""
    ax.annotate(
        "", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.5)
    )


# Title
ax.text(
    50,
    97,
    "TRADING BOT ARCHITECTURE",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color=COLORS["main"],
)
ax.text(
    50,
    94,
    "Multi-Instrument Options Trading System",
    ha="center",
    va="center",
    fontsize=12,
    color=COLORS["subtext"],
)

# ================= MAIN ENTRY POINT =================
draw_box(ax, 40, 82, 20, 6, "Tradebot.py", "Main Entry Point", COLORS["main"], 12)

# ================= CORE MODULES SECTION =================
draw_section(ax, 5, 55, 90, 22, "CORE TRADING MODULES", COLORS["core"])

draw_box(ax, 8, 62, 16, 8, "Scanner", "scanner.py", COLORS["core"])
draw_box(ax, 28, 62, 16, 8, "Manager", "manager.py", COLORS["core"])
draw_box(ax, 48, 62, 18, 8, "Socket Handler", "socket_handler.py", COLORS["core"])
draw_box(ax, 70, 62, 16, 8, "Strategies", "strategies.py", COLORS["core"])

# Add descriptions under core modules
ax.text(
    16,
    59,
    "Signal Detection\nEntry Triggers",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)
ax.text(
    36,
    59,
    "Trail Stop Loss\nP&L Calculation",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)
ax.text(
    57,
    59,
    "WebSocket Feed\nReal-time Data",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)
ax.text(
    78,
    59,
    "EMA Crossover\nRSI + MACD",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)

# ================= SUPPORT MODULES SECTION =================
draw_section(ax, 5, 32, 58, 18, "SUPPORT MODULES", COLORS["support"])

draw_box(ax, 8, 36, 14, 7, "Utils", "utils.py", COLORS["support"])
draw_box(ax, 25, 36, 14, 7, "Heartbeat", "heartbeat.py", COLORS["support"])
draw_box(ax, 42, 36, 18, 7, "Reconciliation", "position_recon.py", COLORS["support"])

ax.text(
    15,
    34,
    "Telegram Alerts\nState I/O",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)
ax.text(
    32,
    34,
    "Dead Man Switch\nMonitoring",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)
ax.text(
    51,
    34,
    "Sync with Broker\nVerify Positions",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)

# ================= NOTIFICATION SECTION (NEW) =================
draw_section(ax, 66, 32, 29, 18, "NOTIFICATIONS (NEW)", COLORS["notification"])

draw_box(ax, 68, 36, 12, 7, "Charts", "chart_gen.py", COLORS["notification"])
draw_box(ax, 82, 36, 12, 7, "EOD Report", "eod_report.py", COLORS["notification"])

ax.text(
    74,
    34,
    "Trade Charts\nTo Telegram",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)
ax.text(
    88,
    34,
    "Daily Summary\nEmail + HTML",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)

# ================= CONFIG SECTION =================
draw_section(ax, 5, 8, 45, 18, "CONFIGURATION", COLORS["config"])

draw_box(ax, 8, 12, 12, 7, "Config", "config.py", COLORS["config"])
draw_box(ax, 23, 12, 14, 7, "Instruments", "instruments.py", COLORS["config"])
draw_box(ax, 40, 12, 8, 7, "State", "stores.py", COLORS["config"])

# ================= DATA SECTION =================
draw_section(ax, 53, 8, 42, 18, "DATA & STORAGE", COLORS["data"])

draw_box(ax, 56, 12, 10, 7, "data/", "", COLORS["data"])
draw_box(ax, 68, 12, 12, 7, "reports/", "", COLORS["data"])
draw_box(ax, 82, 12, 12, 7, "Extras/", "", COLORS["data"])

ax.text(
    61, 10.5, "State\nLogs\nCache", ha="center", fontsize=7, color=COLORS["subtext"]
)
ax.text(
    74,
    10.5,
    "EOD Reports\nHTML Files",
    ha="center",
    fontsize=7,
    color=COLORS["subtext"],
)
ax.text(
    88, 10.5, "Tools\nDocs\nBackup", ha="center", fontsize=7, color=COLORS["subtext"]
)

# ================= EXTERNAL SERVICES =================
ax.text(
    50,
    3,
    "External Services: Dhan API | Telegram Bot | Email (SMTP) | Streamlit Dashboard",
    ha="center",
    fontsize=10,
    color=COLORS["external"],
)

# ================= ARROWS =================
# Main to core modules
draw_arrow(ax, (50, 82), (50, 78))
draw_arrow(ax, (50, 78), (16, 70), COLORS["core"])
draw_arrow(ax, (50, 78), (36, 70), COLORS["core"])
draw_arrow(ax, (50, 78), (57, 70), COLORS["core"])
draw_arrow(ax, (50, 78), (78, 70), COLORS["core"])

# Core to support/notification
draw_arrow(ax, (16, 62), (15, 50), COLORS["support"])
draw_arrow(ax, (36, 62), (32, 50), COLORS["support"])
draw_arrow(ax, (36, 62), (51, 50), COLORS["support"])
draw_arrow(ax, (36, 62), (74, 50), COLORS["notification"])
draw_arrow(ax, (36, 62), (88, 50), COLORS["notification"])

# ================= LEGEND =================
legend_y = 2
legend_items = [
    ("Main Entry", COLORS["main"]),
    ("Core Trading", COLORS["core"]),
    ("Support", COLORS["support"]),
    ("Notifications", COLORS["notification"]),
    ("Config", COLORS["config"]),
    ("Data", COLORS["data"]),
]
legend_x = 5
for label, color in legend_items:
    ax.add_patch(
        plt.Rectangle((legend_x, legend_y - 0.5), 2, 1.5, facecolor=color, alpha=0.7)
    )
    ax.text(legend_x + 3, legend_y + 0.2, label, fontsize=8, color=COLORS["text"])
    legend_x += 14

# Save the figure
plt.tight_layout()
plt.savefig(
    "architecture_diagram.png",
    dpi=150,
    facecolor="#1a1a2e",
    edgecolor="none",
    bbox_inches="tight",
    pad_inches=0.5,
)
print("Architecture diagram saved to architecture_diagram.png")
plt.close()

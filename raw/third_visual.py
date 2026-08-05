import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pathlib import Path


# ============================================================
# FILES
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
BLS_FILE = ROOT / "raw" / "occupation.xlsx"
AEI_FILE = ROOT / "raw" / "aei_job_exposure.csv"
FIG_PATH = ROOT / "figures" / "exposure_vs_bls_projected_growth.png"


AVERAGE_GROWTH = 3.1
EXPOSURE_PERCENTILE = 75

# Raise this to reduce pale yellow points.
COLOR_FLOOR = 0.30

# Lower these values to strengthen red/green separation.
EXPOSURE_POWER = 0.45
GROWTH_POWER = 0.45

POINT_ALPHA = 0.85
FIGURE_SIZE = (10, 6)


bls = pd.read_excel(
    BLS_FILE,
    sheet_name="Table 1.2",
    header=1,
)

aei = pd.read_csv(
    AEI_FILE
)


# Keep detailed occupational rows rather than summary groups.
bls = bls[
    bls["Occupation type"] == "Line item"
].copy()


# Keep only the AI-exposure columns needed for the merge.
aei = aei[
    [
        "occ_code",
        "observed_exposure",
    ]
].copy()


# ============================================================
# MERGE ON SOC CODE
# ============================================================

df = bls.merge(
    aei,
    left_on="2024 National Employment Matrix code",
    right_on="occ_code",
    how="inner",
)


df = df.rename(
    columns={
        "2024 National Employment Matrix title": "occupation",
        "2024 National Employment Matrix code": "soc_code",
        "Employment change, percent, 2024–34": "growth",
        "Employment, 2024": "employment",
        "observed_exposure": "ai_exposure",
    }
)


df = df[
    [
        "occupation",
        "soc_code",
        "growth",
        "employment",
        "ai_exposure",
    ]
].copy()


# Convert plotting columns to numeric values.
for column in [
    "growth",
    "employment",
    "ai_exposure",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


df = df.dropna().reset_index(drop=True)



# Normalize AI exposure between 0 and 1.
exposure_min = df["ai_exposure"].min()
exposure_max = df["ai_exposure"].max()

df["exposure_norm"] = (
    df["ai_exposure"] - exposure_min
) / (
    exposure_max - exposure_min
)


# Calculate distance above or below average projected growth.
df["growth_difference"] = (
    df["growth"] - AVERAGE_GROWTH
)


max_growth_distance = (
    df["growth_difference"]
    .abs()
    .max()
)


df["growth_magnitude"] = (
    df["growth_difference"].abs()
    / max_growth_distance
)


# Positive values indicate above-average growth.
# Negative values indicate below-average growth.
df["growth_sign"] = np.sign(
    df["growth_difference"]
)


# Negative values become red.
# Positive values become green.
# Values close to zero become yellow or orange.
df["color_score"] = (
    df["growth_sign"]
    * (
        COLOR_FLOOR
        + (
            1 - COLOR_FLOOR
        )
        * df["exposure_norm"].pow(
            EXPOSURE_POWER
        )
    )
    * df["growth_magnitude"].pow(
        GROWTH_POWER
    )
)


df["color_score"] = (
    df["color_score"]
    .clip(-1, 1)
)


# ============================================================
# COLOR SCALE
# ============================================================

cmap = LinearSegmentedColormap.from_list(
    "red_yellow_green",
    [
        (0.00, "#a50026"),
        (0.20, "#d73027"),
        (0.38, "#f46d43"),
        (0.50, "#ffe66d"),
        (0.62, "#a6d96a"),
        (0.80, "#1a9850"),
        (1.00, "#006837"),
    ],
)


norm = TwoSlopeNorm(
    vmin=-1,
    vcenter=0,
    vmax=1,
)



exposure_cutoff = np.percentile(
    df["ai_exposure"],
    EXPOSURE_PERCENTILE,
)


# Employment is reported in thousands.
# Log scaling prevents large occupations from dominating.
bubble_sizes = (
    12
    + 14
    * np.log10(
        df["employment"] + 1
    )
)

fig = plt.figure(
    figsize=FIGURE_SIZE
)


# Fixed axes positions prevent the horizontal y-axis label
# and colorbar labels from being cut off.
#
# Format:
# [left, bottom, width, height]
ax = fig.add_axes(
    [0.17, 0.29, 0.80, 0.60]
)


colorbar_ax = fig.add_axes(
    [0.21, 0.095, 0.72, 0.025]
)


# ============================================================
# SCATTERPLOT
# ============================================================

scatter = ax.scatter(
    df["ai_exposure"],
    df["growth"],
    c=df["color_score"],
    cmap=cmap,
    norm=norm,
    s=bubble_sizes,
    alpha=POINT_ALPHA,
    edgecolors="white",
    linewidths=0.25,
)


# Hide the default x/y coordinate readout in the toolbar.
ax.format_coord = lambda x, y: ""


# Create one annotation and reuse it for whichever point
# the mouse is currently hovering over.
hover_annotation = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=8,
    ha="left",
    va="bottom",
    annotation_clip=False,
    bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "black",
        "linewidth": 0.7,
        "alpha": 0.95,
    },
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 0.7,
        "alpha": 0.75,
    },
)

hover_annotation.set_visible(False)


def get_nearest_point_index(event, candidate_indices):
    """
    When several points overlap, return the point nearest
    to the mouse cursor in display coordinates.
    """

    offsets = scatter.get_offsets()

    candidate_points = offsets[
        candidate_indices
    ]

    display_points = ax.transData.transform(
        candidate_points
    )

    mouse_position = np.array(
        [
            event.x,
            event.y,
        ]
    )

    distances = np.linalg.norm(
        display_points - mouse_position,
        axis=1,
    )

    nearest_position = np.argmin(
        distances
    )

    return int(
        candidate_indices[nearest_position]
    )


def update_hover_annotation(point_index):
    """
    Update the hover label with the occupation associated
    with the selected point.
    """

    point = scatter.get_offsets()[
        point_index
    ]

    row = df.iloc[
        point_index
    ]

    hover_annotation.xy = point

    # Display only the occupation title.
    hover_annotation.set_text(
        row["occupation"]
    )


    # Place the label on the side with more available room.
    x_middle = (
        ax.get_xlim()[0]
        + ax.get_xlim()[1]
    ) / 2

    y_middle = (
        ax.get_ylim()[0]
        + ax.get_ylim()[1]
    ) / 2


    horizontal_offset = (
        -10
        if point[0] > x_middle
        else 10
    )

    vertical_offset = (
        -10
        if point[1] > y_middle
        else 10
    )


    hover_annotation.set_position(
        (
            horizontal_offset,
            vertical_offset,
        )
    )


    hover_annotation.set_ha(
        "right"
        if horizontal_offset < 0
        else "left"
    )


    hover_annotation.set_va(
        "top"
        if vertical_offset < 0
        else "bottom"
    )


def on_hover(event):
    """
    Show an occupation title while hovering over a point.
    Hide the label when the mouse moves away.
    """

    if event.inaxes != ax:

        if hover_annotation.get_visible():

            hover_annotation.set_visible(
                False
            )

            fig.canvas.draw_idle()

        return


    contains_point, point_information = (
        scatter.contains(event)
    )


    if contains_point:

        candidate_indices = (
            point_information["ind"]
        )

        point_index = get_nearest_point_index(
            event,
            candidate_indices,
        )

        update_hover_annotation(
            point_index
        )

        hover_annotation.set_visible(
            True
        )

        fig.canvas.draw_idle()

    else:

        if hover_annotation.get_visible():

            hover_annotation.set_visible(
                False
            )

            fig.canvas.draw_idle()


# Connect mouse movement to the hover function.
fig.canvas.mpl_connect(
    "motion_notify_event",
    on_hover,
)



ax.axhline(
    AVERAGE_GROWTH,
    color="black",
    linewidth=0.9,
)


ax.axvline(
    exposure_cutoff,
    color="black",
    linewidth=0.9,
    linestyle="--",
)


ax.set_title(
    "AI exposure versus BLS projected occupational growth",
    loc="left",
    fontsize=13,
    fontweight="semibold",
    pad=18,
)


ax.text(
    0,
    1.015,
    (
        "Observed Claude exposure matched to "
        f"2024–34 BLS projections • {len(df)} occupations"
    ),
    transform=ax.transAxes,
    fontsize=8,
    va="bottom",
)



ax.set_xlabel(
    "Observed AI exposure",
    fontsize=9,
    labelpad=6,
)


# Figure-level text keeps the horizontal y-axis label
# fully inside the 10 × 6 canvas.
fig.text(
    0.018,
    0.585,
    "Projected employment\nchange, 2024–34 (%)",
    fontsize=9,
    ha="left",
    va="center",
)


ax.tick_params(
    axis="both",
    labelsize=8,
)


ax.grid(
    alpha=0.18,
    linewidth=0.6,
)


x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()


ax.text(
    x_max * 0.82,
    y_max * 0.84,
    "Higher exposure\nAbove-average growth",
    ha="center",
    va="center",
    fontsize=8,
)


ax.text(
    x_max * 0.82,
    y_min * 0.76,
    "Higher exposure\nBelow-average growth",
    ha="center",
    va="center",
    fontsize=8,
)


ax.text(
    exposure_cutoff * 0.40,
    y_max * 0.84,
    "Lower exposure\nAbove-average growth",
    ha="center",
    va="center",
    fontsize=8,
)


ax.text(
    exposure_cutoff * 0.40,
    y_min * 0.76,
    "Lower exposure\nBelow-average growth",
    ha="center",
    va="center",
    fontsize=8,
)


ax.text(
    exposure_cutoff,
    y_max * 0.97,
    f"  {EXPOSURE_PERCENTILE}th percentile exposure",
    va="top",
    fontsize=8,
)


ax.text(
    x_min + 0.004,
    AVERAGE_GROWTH + 0.7,
    f"All-occupation growth: {AVERAGE_GROWTH:.1f}%",
    fontsize=8,
)

colorbar = fig.colorbar(
    scatter,
    cax=colorbar_ax,
    orientation="horizontal",
)


colorbar.set_ticks(
    [
        -1,
        0,
        1,
    ]
)


colorbar.set_ticklabels(
    [
        "Higher exposure +\nbelow-average growth",
        "Neutral /\nin-between",
        "Higher exposure +\nabove-average growth",
    ]
)


colorbar.ax.tick_params(
    labelsize=8,
    pad=3,
)


colorbar.set_label(
    "Combined AI exposure and projected-growth signal",
    fontsize=8,
    labelpad=5,
)


# Put the colorbar title above the bar.
colorbar.ax.xaxis.set_label_position(
    "top"
)


# ============================================================
# DISPLAY
# ============================================================

# Do not use tight_layout() or constrained_layout().
# Fixed axes positions are already controlling the spacing.
FIG_PATH.parent.mkdir(exist_ok=True)
plt.savefig(FIG_PATH, dpi=200)
plt.close(fig)


print(f"Matched occupations: {len(df)}")
print(f"Exposure cutoff: {exposure_cutoff:.4f}")
print(f"Saved: {FIG_PATH}")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SKILLS_XLSX = "skills.xlsx"          # both files in the same folder you run from
OCC_XLSX = "occupation.xlsx"
FIG_PATH = "AI_Usage_vs_Labor.png"

SKILLS = ["Critical and analytical thinking", "Mechanical",
          "Problem solving and decision making", "Computers and information technology",
          "Detail oriented", "Physical strength and stamina",
          "Interpersonal", "Customer service"]

BLUE, GREEN = "#2563eb", "#16a34a"                  # AI used more / AI used less

# ----------------------------------------------------------------------
# Load skill percentiles (Table 6.5) and AI grouping (Table 1.12)
# ----------------------------------------------------------------------
t65 = pd.read_excel(SKILLS_XLSX, sheet_name="Table 6.5", header=1)
t65 = t65.rename(columns={"2024 National Employment Matrix code": "occ_code"})

t112 = pd.read_excel(OCC_XLSX, sheet_name="Table 1.12", header=1)
t112.columns = ["occ_title", "occ_code", "ind_title", "ind_code", "factor"]
t112 = t112.dropna(subset=["factor"])

ai = t112[t112["factor"].str.contains(r"\bAI\b", regex=True)].copy()
ai["direction"] = ai["factor"].str.extract(r"share (increases|decreases)", expand=False)
per_occ = ai.groupby("occ_code")["direction"].agg(lambda s: tuple(sorted(set(s.dropna()))))

codes_more = per_occ[per_occ == ("decreases",)].index   # AI is used more
codes_less = per_occ[per_occ == ("increases",)].index   # AI is used less

med_more = t65[t65["occ_code"].isin(codes_more)][SKILLS].median()
med_less = t65[t65["occ_code"].isin(codes_less)][SKILLS].median()

# sort rows by gap (AI-less minus AI-more), largest gap on top
order = (med_less - med_more).sort_values(ascending=False, kind="stable").index
med_more, med_less = med_more[order], med_less[order]

# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(14.5, 8.5))

ys = range(len(order))
ax.axvline(50, color="#e8e8e8", lw=1.6, zorder=0)
for y, sk in zip(ys, order):
    a, b = med_more[sk], med_less[sk]
    ax.plot([a, b], [y, y], color="#dddddd", lw=3, solid_capstyle="round", zorder=1)
    ax.plot(a, y, "o", color=BLUE, ms=13, zorder=3)
    ax.plot(b, y, "o", color=GREEN, ms=13, zorder=3)

ax.set_yticks(list(ys))
ax.set_yticklabels(order, fontsize=15)
ax.set_ylim(7.55, -0.8)                              # first row on top
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 20))
ax.tick_params(axis="x", labelsize=15, length=7, color="black")
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Median skill percentile of the group", fontsize=16)

for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")

ax.set_title("AI Usage vs type of Labor", fontsize=24, fontweight="bold",
             loc="left", pad=28, x=-0.28)

legend_handles = [
    Line2D([], [], marker="o", color=BLUE, linestyle="None", ms=15, label="AI is used more"),
    Line2D([], [], marker="o", color=GREEN, linestyle="None", ms=15, label="AI is used less"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=False,
          fontsize=15, handletextpad=0.4, borderaxespad=0.2)

plt.subplots_adjust(left=0.29, right=0.985, top=0.87, bottom=0.12)
plt.savefig(FIG_PATH, dpi=200)
print(f"Saved: {FIG_PATH}")
plt.show()
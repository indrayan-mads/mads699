import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XLSX_PATH = ROOT / "raw" / "occupation.xlsx"   # BLS sheet "Table 1.1"
FIG_PATH = ROOT / "figures" / "employment_projections_slope.png"

#data loading
raw = pd.read_excel(XLSX_PATH, sheet_name="Table 1.1", header=1)
raw.columns = ["title", "soc", "emp2024", "emp2034", "chg_num", "chg_pct", "wage"]

# keep only major-group rows (SOC xx-0000), drop the "Total, all occupations" line
soc = raw["soc"].astype(str)
df = raw[(soc.str.len() == 7) & soc.str.endswith("-0000")].copy()
df = df[df["soc"] != "00-0000"]

df["name"] = df["title"].str.strip().str.removesuffix(" occupations")
df["v2024"] = df["emp2024"] / 1000.0          # thousands -> millions
df["v2034"] = df["emp2034"] / 1000.0

# color rule: green = growing >= 6% | red = declining | gray = modest change
def category(pct):
    if pct >= 6:
        return "green"
    if pct < 0:
        return "red"
    return "gray"

df["cat"] = df["chg_pct"].apply(category)
df = df.sort_values("v2024", ascending=False).reset_index(drop=True)

COLORS = {"green": "#1a7d5a", "red": "#c9455b", "gray": "#8f8f88"}
LINE_GRAY = "#cccbc4"   # light warm gray for modest-change lines


#Label spreading and adjustments for visual appeasement
def spread(targets, min_gap, lo, hi, iters=600):
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    p = [targets[i] for i in order] 
    for _ in range(iters):
        moved = False
        for i in range(1, len(p)):
            d = p[i] - p[i - 1]
            if d < min_gap:
                shift = (min_gap - d) / 2
                p[i - 1] -= shift
                p[i] += shift
                moved = True
        p[0] = max(p[0], lo)
        p[-1] = min(p[-1], hi)
        if not moved:
            break
    out = [0.0] * len(targets)
    for k, i in enumerate(order):
        out[i] = p[k]
    return out

label_y = spread(df["v2034"].tolist(), min_gap=0.62,
                 lo=df["v2034"].min(), hi=df["v2034"].max())


#Graph
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(14.5, 11.5))

X0, X1 = 0.0, 1.0
LABEL_X = 1.11          # where label text begins
LEAD_END = 1.08         # where the leader line meets the text

for row, ly in zip(df.itertuples(), label_y):
    cat = row.cat
    if cat == "gray":
        lcol, tcol, lw, ms, weight, z = LINE_GRAY, COLORS["gray"], 1.4, 5, "normal", 2
    else:
        lcol = tcol = COLORS[cat]
        lw, ms, weight, z = 3.1, 9, "bold", 4

    # the slope line + endpoint dots (plotted at true, unrounded values)
    ax.plot([X0, X1], [row.v2024, row.v2034], color=lcol, lw=lw,
            solid_capstyle="round", zorder=z)
    ax.plot([X0, X1], [row.v2024, row.v2034], "o", color=lcol, ms=ms, zorder=z + 1)

    # leader line from the 2034 dot out to the (de-conflicted) label
    ax.plot([X1, LEAD_END], [row.v2034, ly], color=lcol, lw=0.8,
            alpha=0.45 if cat != "gray" else 0.9, zorder=1)

    # label text — values rounded for display only
    label = (f"{row.name}   {row.v2024:.1f} \u2192 {row.v2034:.1f}M"
             f"  ({row.chg_pct:+.1f}%)")
    ax.text(LABEL_X, ly, label, color=tcol, fontsize=11,
            fontweight=weight, va="center", ha="left")

#more cosmetics to for cleaner look
ax.set_xlim(-0.04, 3.55)
ax.set_ylim(-0.4, 19.6)

yt = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]
ax.set_yticks(yt)
ax.set_yticklabels([f"{v:.1f}" for v in yt], fontsize=11, color="#555")
for y in yt:
    ax.axhline(y, color="#eeece7", lw=1, zorder=0)

ax.set_xticks([X0, X1])
ax.set_xticklabels(["2024", "2034\n(projected)"], fontsize=13, color="#333")
ax.tick_params(length=0)
ax.set_ylabel("Employment (millions)", fontsize=12.5, color="#444")

for s in ax.spines.values():
    s.set_visible(False)

# ----- titles & source -----
ax.set_title("Where US employment is heading, 2024 \u2192 2034",
             fontsize=25, fontweight="bold", loc="left", pad=34, color="#1a1a1a")
ax.text(-0.04, 1.028,
        "Major occupational groups  \u00b7  green = growing \u2265 6%  \u00b7  "
        "red = declining  \u00b7  gray = modest change",
        transform=ax.transAxes, fontsize=13, color="#777")
ax.text(-0.04, -0.075,
        "Source: US Bureau of Labor Statistics, Employment Projections program (Table 1.1)",
        transform=ax.transAxes, fontsize=11, color="#999")

plt.subplots_adjust(left=0.055, right=0.99, top=0.9, bottom=0.09)
FIG_PATH.parent.mkdir(exist_ok=True)
plt.savefig(FIG_PATH, dpi=170)
plt.close(fig)
print(f"Saved: {FIG_PATH}")

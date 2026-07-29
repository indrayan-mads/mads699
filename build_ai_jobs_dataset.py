"""Build the occupation-level AI exposure / BLS projection-revision dataset.

Joins two BLS employment-projection vintages (one finalized before generative AI,
one after) to four independent measures of AI exposure and the O*NET task
inventory, producing one occupation-level panel and one occupation x task file.

Usage:
    python build_ai_jobs_dataset.py                 # defaults: 2021-31 -> 2024-34
    python build_ai_jobs_dataset.py --pre 2019-29   # alternative pre-AI vintage
    python build_ai_jobs_dataset.py --manual        # list files to download by hand

All required inputs are committed under raw/, so a fresh clone runs offline.
Network fetching only happens for files that are missing from that directory.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("Missing dependency. Run: pip install -r requirements.txt")

# Paths are anchored to this file, not the working directory, so the scripts
# behave the same whether invoked from the repo root or from anywhere else.
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "out"

# Every input file is committed to raw/, so these requests normally never fire.
# They exist as a fallback for a clone that deletes raw/, and to document
# provenance. We identify the project honestly rather than impersonating a
# browser; if a host declines the request, use --manual instead.
USER_AGENT = "mads699-capstone/1.0 (academic research; contact via repo issues)"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})

REQUEST_TIMEOUT = 120  # seconds

ONET_RE = re.compile(r"^\d{2}-\d{4}\.\d{2}$")

############# Source URLs #############

ARCHIVE_URL = "https://www.bls.gov/emp/projections-archive/{v}.zip"
CURRENT_OCC_URL = "https://www.bls.gov/emp/ind-occ-matrix/occupation.xlsx"
CURRENT_VINTAGE = "2024-34"  # update when BLS publishes a new cycle

AIOE_URLS = [
    "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx",
    "https://raw.githubusercontent.com/AIOE-Data/AIOE/master/AIOE_DataAppendix.xlsx",
]
CROSSWALK_URLS = [
    "https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx",
    "https://www.bls.gov/soc/soc_2010_to_2018_crosswalk.xlsx",
]
GPTS_BASE = "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/"
HF = "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/"

# O*NET download URLs are version-numbered. The committed copy in raw/ pins the
# release actually used in the analysis; these candidates are only a fallback,
# newest first, and a different version may shift task counts slightly.
ONET_VERSIONS = ["30_3", "30_2", "30_1", "30_0", "29_3", "29_2", "29_1", "29_0", "28_3"]
ONET_URL = "https://www.onetcenter.org/dl_files/database/db_{v}_excel/{f}"

# File names and excel sheets. Printed by --manual.
MANUAL_SOURCES = [
    ("occupation_2024-34.xlsx", "https://www.bls.gov/emp/ind-occ-matrix/occupation.xlsx",
     "current-cycle occupation workbook (Table 1.2)"),
    ("occupation_2021-31.xlsx", "https://www.bls.gov/emp/data/projections-archive.htm",
     "open the 2021-31 archive zip, extract occupation.xlsx, rename as shown"),
    ("AIOE_DataAppendix.xlsx", "https://github.com/AIOE-Data/AIOE",
     "Felten/Raj/Seamans AI Occupational Exposure appendix"),
    ("soc_2010_to_2018_crosswalk.xlsx", "https://www.bls.gov/soc/2018/",
     "optional; without it the build assumes SOC codes are unchanged"),
    ("gpts_occ_level.csv", GPTS_BASE + "occ_level.csv",
     "Eloundou et al. occupation-level exposure"),
    ("gpts_full_labelset.tsv", GPTS_BASE + "full_labelset.tsv",
     "Eloundou et al. task-level exposure labels"),
    ("onet_task_statements.xlsx", "https://www.onetcenter.org/database.html",
     "O*NET 'Task Statements' Excel file (required)"),
    ("onet_task_ratings.xlsx", "https://www.onetcenter.org/database.html",
     "O*NET 'Task Ratings' Excel file (optional; supplies importance scores)"),
    ("aei_job_exposure.csv",
     "https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main",
     "job_exposure.csv from the labor-market-impacts folder"),
    ("aei_task_pct.csv",
     "https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main",
     "task_pct csv from the same dataset"),
]


# ----------------------------------------------------------------------------
# Small utilities - For readability and debugging
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[build] {msg}")


def warn(msg: str) -> None:
    print(f"[build][WARN] {msg}")


#Session initalization and Requests
def download(url: str, dest: Path, required: bool = False, note: str = "") -> Path | None:
    """Download url -> dest with caching. Returns dest, or None on failure."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"cached: {dest.name}")
        return dest
    log(f"downloading {url}")
    try:
        r = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except Exception as e:  # noqa: BLE001 - any network/parse failure is non-fatal here
        msg = (f"could not fetch {url} ({e}). {note} "
               f"Download it manually to {dest} and re-run, or use --manual.")
        if required:
            sys.exit(f"[build][FATAL] {msg}")
        warn(msg)
        return None


def try_urls(urls: list[str], dest: Path, required: bool = False,
             note: str = "") -> Path | None:
    """Try candidate URLs in order until one works (handles moved files)."""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"cached: {dest.name}")
        return dest
    for u in urls:
        p = download(u, dest)
        if p is not None:
            return p
    msg = (f"none of the candidate URLs worked for {dest.name}. {note} "
           f"Download manually to {dest} and re-run.")
    if required:
        sys.exit(f"[build][FATAL] {msg}")
    warn(msg)
    return None


#Data Cleaning and Organization
def find_col(df: pd.DataFrame, *patterns: str) -> str | None:
    """Return the first column whose name matches any regex (case-insensitive)."""
    for pat in patterns:
        for c in df.columns:
            if re.search(pat, str(c), flags=re.I):
                return str(c)
    return None


def clean_soc(x) -> str | None:
    """Normalize an SOC-like code: strip footnote markers and whitespace."""
    if pd.isna(x):
        return None
    s = re.sub(r"[^\d\-.]", "", str(x)).strip()
    m = re.match(r"^(\d{2}-\d{4})", s)
    return m.group(1) if m else None


def clean_onet(x) -> str | None:
    """Keep only well-formed O*NET-SOC codes (e.g. 15-1252.00)."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if ONET_RE.match(s) else None


def to_num(s: pd.Series) -> pd.Series:
    """Coerce BLS-style numbers: em-dashes, '$239,200+', '>=$239,200', footnotes."""
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r"[\$,]", "", regex=True)
         .str.replace(r"[\u2265>+=]", "", regex=True)
         .str.replace(r"^\s*[\u2014\u2013-]\s*$", "", regex=True)
         .str.strip(),
        errors="coerce",
    )


def norm_text(s) -> str:
    """Normalize task text so the task-level joins can use exact matching."""
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower().rstrip(".")


# ----------------------------------------------------------------------------
# 1) BLS Employment Projections (National Employment Matrix, Table 1.2)
# ----------------------------------------------------------------------------

def vintage_years(vintage: str) -> tuple[int, int]:
    """'2021-31' -> (2021, 2031)."""
    base = int(vintage.split("-")[0])
    return base, base + 10


def load_bls_vintage(vintage: str) -> pd.DataFrame:
    """Return tidy Table 1.2 for one projection vintage, keyed by soc_code."""
    base, proj = vintage_years(vintage)
    if vintage == CURRENT_VINTAGE:
        xlsx = download(CURRENT_OCC_URL, RAW / f"occupation_{vintage}.xlsx", required=True,
                        note="This is the current-cycle occupation workbook (bls.gov/emp).")
    else:
        # Prefer an already-extracted workbook: it is ~470 KB versus a ~9 MB zip,
        # so the repo ships the xlsx alone and skips the archive entirely.
        xlsx = RAW / f"occupation_{vintage}.xlsx"
        if not xlsx.exists():
            z = download(ARCHIVE_URL.format(v=vintage), RAW / f"bls_{vintage}.zip",
                         required=True,
                         note="Archive index: https://www.bls.gov/emp/data/projections-archive.htm")
            with zipfile.ZipFile(z) as zf:
                cands = [n for n in zf.namelist()
                         if re.search(r"occupation.*\.xlsx$", n, re.I)]
                if not cands:
                    sys.exit(f"[build][FATAL] no occupation*.xlsx inside {z.name}; "
                             f"inspect the zip and extract Table 1.2 manually.")
                cands.sort(key=len)  # prefer the plain 'occupation.xlsx'
                xlsx.write_bytes(zf.read(cands[0]))

    xl = pd.ExcelFile(xlsx)
    sheet = next((s for s in xl.sheet_names if "1.2" in s), xl.sheet_names[0])
    # Row 0 is the table title; row 1 holds the header in every recent vintage.
    df = pd.read_excel(xlsx, sheet_name=sheet, skiprows=1)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    c_title = find_col(df, r"matrix title", r"occupation title", r"title")
    c_code = find_col(df, r"matrix code", r"occupation code", r"\bcode\b")
    c_type = find_col(df, r"occupation type")
    c_emp_b = find_col(df, rf"employment,?\s*{base}\b")
    c_emp_p = find_col(df, rf"employment,?\s*{proj}\b")
    
    # Notes: Header wording differs across vintages: older files say "Percent employment
    # change"; the 2024-34 workbook says "Employment change, percent, 2024-34".
    
    c_pct = find_col(df, r"percent employment change", r"employment change,?\s*percent")
    c_open = find_col(df, r"occupational openings")
    c_wage = find_col(df, r"median annual wage")
    c_edu = find_col(df, r"typical entry-?level education", r"typical education needed")

    missing = [n for n, c in [("code", c_code), ("emp_base", c_emp_b),
                              ("emp_proj", c_emp_p), ("pct_change", c_pct)] if c is None]
    if missing:
        sys.exit(f"[build][FATAL] vintage {vintage}: could not locate columns {missing}. "
                 f"Headers seen: {list(df.columns)}")

    out = pd.DataFrame({
        "soc_code": df[c_code].map(clean_soc),
        "occ_title": df[c_title].astype(str).str.strip() if c_title else None,
        "emp_base": to_num(df[c_emp_b]),
        "emp_proj": to_num(df[c_emp_p]),
        "pct_change": to_num(df[c_pct]),
    })
    if c_open:
        out["annual_openings"] = to_num(df[c_open])
    if c_wage:
        out["median_wage"] = to_num(df[c_wage])
    if c_edu:
        out["typical_education"] = df[c_edu].astype(str).str.strip()
    if c_type:
        # Keep detailed occupations, drop summary/rollup lines. `out` is built
        # from `df`, so it carries df's index and this mask aligns.
        out = out[df[c_type].astype(str).str.contains("line item", case=False, na=False)]
    out = out[out["soc_code"].notna() & (out["soc_code"] != "00-0000")]
    out = out.drop_duplicates(subset="soc_code")

    b, p = str(base), str(proj)
    out = out.rename(columns={
        "emp_base": f"emp_{b}", "emp_proj": f"emp_{p}",
        "pct_change": f"pct_chg_{b}_{p}",
        "annual_openings": f"openings_{b}_{p}",
        "median_wage": f"median_wage_{b}",
        "typical_education": f"education_{b}",
    })
    log(f"BLS {vintage}: {len(out)} detailed occupations")
    return out


# ----------------------------------------------------------------------------
# 2) Felten/Raj/Seamans AIOE  (SOC 2010 -> crosswalk to SOC 2018)
# ----------------------------------------------------------------------------

def load_felten() -> pd.DataFrame | None:
    p = try_urls(AIOE_URLS, RAW / "AIOE_DataAppendix.xlsx",
                 note="Repo: https://github.com/AIOE-Data/AIOE (AIOE_DataAppendix.xlsx).")
    if p is None:
        return None
    xl = pd.ExcelFile(p)
    sheet = next((s for s in xl.sheet_names if re.search(r"appendix a|occupation", s, re.I)),
                 xl.sheet_names[0])
    df = pd.read_excel(p, sheet_name=sheet)
    c_code = find_col(df, r"soc.*code", r"\bcode\b")
    c_aioe = find_col(df, r"\baioe\b", r"exposure")
    if not (c_code and c_aioe):
        warn(f"Felten AIOE: no code/score columns in sheet '{sheet}'; skipping.")
        return None
    df = pd.DataFrame({"soc2010": df[c_code].map(clean_soc),
                       "aioe_felten": pd.to_numeric(df[c_aioe], errors="coerce")})
    df = df.dropna(subset=["soc2010"])

    
    # SOC 2010 -> 2018 crosswalk. Most codes are unchanged; the crosswalk fixes
    # the recoded minority. Falling back costs accuracy on those occupations.
    xw_path = try_urls(CROSSWALK_URLS, RAW / "soc_2010_to_2018_crosswalk.xlsx",
                       note="See https://www.bls.gov/soc/2018/ for crosswalk files.")
    if xw_path is not None:
        xw_raw = pd.read_excel(xw_path, header=None)
        hdr = next((i for i in range(min(12, len(xw_raw)))
                    if xw_raw.iloc[i].astype(str).str.contains("2010", case=False).any()
                    and xw_raw.iloc[i].astype(str).str.contains("2018", case=False).any()), None)
        if hdr is not None:
            xw = pd.read_excel(xw_path, header=hdr)
            c10 = find_col(xw, r"2010.*code")
            c18 = find_col(xw, r"2018.*code")
            if c10 and c18:
                xw = pd.DataFrame({"soc2010": xw[c10].map(clean_soc),
                                   "soc_code": xw[c18].map(clean_soc)}).dropna()
                df = df.merge(xw, on="soc2010", how="left")
    if "soc_code" not in df.columns:
        warn("Felten AIOE: crosswalk unavailable; assuming SOC codes unchanged 2010->2018. "
             "This is a known accuracy limitation, reported in the README caveats.")
        df["soc_code"] = df["soc2010"]
    df["soc_code"] = df["soc_code"].fillna(df["soc2010"])
    out = df.groupby("soc_code", as_index=False)["aioe_felten"].mean()
    log(f"Felten AIOE: {len(out)} SOC-2018 occupations")
    return out


# ----------------------------------------------------------------------------
# 3) Eloundou et al. (OpenAI "GPTs are GPTs") - occupation and task level
# ----------------------------------------------------------------------------

def load_eloundou_occ() -> pd.DataFrame | None:
    p = download(GPTS_BASE + "occ_level.csv", RAW / "gpts_occ_level.csv",
                 note="Repo: https://github.com/openai/GPTs-are-GPTs (data/occ_level.csv).")
    if p is None:
        return None
    df = pd.read_csv(p)
    c_code = find_col(df, r"o\*?net.*code", r"soc.*code", r"^code$", r"occ.*code")
    if c_code is None:
        warn(f"Eloundou occ_level: no code column (cols={list(df.columns)}); skipping.")
        return None
        
    # O*NET-SOC codes are the 2018 SOC code plus a .XX detail suffix.
    codes = df[c_code].astype(str).str.strip()
    df["soc_code"] = codes.str.slice(0, 7).where(codes.str.match(r"\d{2}-\d{4}"), None)
    score_cols = [c for c in df.columns
                  if re.search(r"alpha|beta|gamma|zeta|expos", str(c), re.I)
                  and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not score_cols:
        warn("Eloundou occ_level: no exposure score columns detected; skipping.")
        return None
    keep = df[["soc_code"] + score_cols].dropna(subset=["soc_code"])
    keep[score_cols] = keep[score_cols].apply(pd.to_numeric, errors="coerce")
    out = keep.groupby("soc_code", as_index=False)[score_cols].mean()
    ren = {c: "gpts_" + re.sub(r"\W+", "_", str(c)).strip("_").lower() for c in score_cols}
    out = out.rename(columns=ren)
    log(f"Eloundou occupation exposure: {len(out)} occupations, cols={list(ren.values())}")
    return out


def load_eloundou_tasks() -> pd.DataFrame | None:
    p = download(GPTS_BASE + "full_labelset.tsv", RAW / "gpts_full_labelset.tsv",
                 note="Task-level exposure labels from github.com/openai/GPTs-are-GPTs.")
    if p is None:
        return None
    df = pd.read_csv(p, sep="\t")
    c_task = find_col(df, r"^task$", r"task.*(statement|text|description)")
    c_code = find_col(df, r"o\*?net.*code", r"soc.*code")
    if c_task is None:
        warn(f"Eloundou task labels: no task text column (cols={list(df.columns)}); skipping.")
        return None
    label_cols = [c for c in df.columns
                  if re.search(r"alpha|beta|gamma|zeta|label|gpt4|human|expos", str(c), re.I)
                  and c not in (c_task, c_code)]
    out = pd.DataFrame({"task_norm": df[c_task].map(norm_text)})
    if c_code is not None:
        out["onet_code"] = df[c_code].map(clean_onet)
    for c in label_cols:
        out["gpts_task_" + re.sub(r"\W+", "_", str(c)).strip("_").lower()] = df[c]
    keys = [k for k in ["onet_code", "task_norm"] if k in out.columns]
    out = out.drop_duplicates(subset=keys)
    log(f"Eloundou task labels: {len(out)} rows, {len(label_cols)} label cols")
    return out


# ----------------------------------------------------------------------------
# 4) O*NET task statements + importance ratings
# ----------------------------------------------------------------------------

def load_onet_tasks() -> pd.DataFrame:
    stmts = try_urls([ONET_URL.format(v=v, f="Task%20Statements.xlsx") for v in ONET_VERSIONS],
                     RAW / "onet_task_statements.xlsx", required=True,
                     note="Get 'Task Statements.xlsx' from https://www.onetcenter.org/database.html")
    ratings = try_urls([ONET_URL.format(v=v, f="Task%20Ratings.xlsx") for v in ONET_VERSIONS],
                       RAW / "onet_task_ratings.xlsx",
                       note="Get 'Task Ratings.xlsx' from https://www.onetcenter.org/database.html")

    df = pd.read_excel(stmts)
    t = pd.DataFrame({
        "onet_code": df[find_col(df, r"o\*?net.*code")].map(clean_onet),
        "onet_title": df[find_col(df, r"^title$")].astype(str).str.strip(),
        "task_id": df[find_col(df, r"task id")],
        "task": df[find_col(df, r"^task$")].astype(str).str.strip(),
        "task_type": df[find_col(df, r"task type")],
    })
    t["task_norm"] = t["task"].map(norm_text)
    t["soc_code"] = t["onet_code"].str.slice(0, 7)  # O*NET-SOC 2019 -> 2018 SOC prefix

    if ratings is not None:
        r = pd.read_excel(ratings)
        c_scale = find_col(r, r"scale id")
        c_val = find_col(r, r"data value")
        c_tid = find_col(r, r"task id")
        if all([c_scale, c_val, c_tid]):
            # Scale 'IM' is the importance rating; other scales (frequency,
            # relevance) are dropped.
            imp = (r[r[c_scale].astype(str).str.upper() == "IM"]
                   .groupby(c_tid, as_index=False)[c_val].mean()
                   .rename(columns={c_tid: "task_id", c_val: "task_importance"}))
            t = t.merge(imp, on="task_id", how="left")
    log(f"O*NET tasks: {len(t)} task rows across {t['onet_code'].nunique()} O*NET occupations")
    return t


# ----------------------------------------------------------------------------
# 5) Anthropic Economic Index - observed Claude usage by task and occupation
# ----------------------------------------------------------------------------

def load_aei_tasks() -> pd.DataFrame | None:
    """Task-level share of Claude conversations mapped to each O*NET task."""
    p = try_urls(
        [HF + "release_2025_03_27/task_pct_v2.csv",
         HF + "release_2025_03_27/task_pct_v1.csv",
         HF + "task_pct_v2.csv"],
        RAW / "aei_task_pct.csv",
        note="Browse https://huggingface.co/datasets/Anthropic/EconomicIndex for task_pct files.")
    if p is None:
        return None
    df = pd.read_csv(p)
    c_task = find_col(df, r"task")
    c_pct = find_col(df, r"pct|percent|share")
    if not (c_task and c_pct):
        warn(f"AEI task_pct: unexpected columns {list(df.columns)}; skipping.")
        return None
    out = pd.DataFrame({"task_norm": df[c_task].map(norm_text),
                        "aei_claude_task_pct": pd.to_numeric(df[c_pct], errors="coerce")})
    out = out.groupby("task_norm", as_index=False)["aei_claude_task_pct"].sum()
    log(f"AEI task usage: {len(out)} tasks with Claude-usage share")
    return out


def load_aei_job_exposure() -> pd.DataFrame | None:
    """Occupation-level AI exposure from the AEI labor-market-impacts release."""
    cands = []
    for folder in ["labor_market_impacts", "labor-market-impacts",
                   "release_labor_market_impacts", "labor_market"]:
        cands.append(HF + f"{folder}/job_exposure.csv")
        cands.append(HF + f"{folder}/data/job_exposure.csv")
    p = try_urls(cands, RAW / "aei_job_exposure.csv",
                 note="Find job_exposure.csv in the 'Labor market impacts' folder at "
                      "https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main "
                      "and save it as raw/aei_job_exposure.csv.")
    if p is None:
        return None
    df = pd.read_csv(p)
    c_code = find_col(df, r"o\*?net.*code", r"soc.*code", r"occ.*code", r"^code$")
    if c_code is None:
        warn(f"AEI job_exposure: no code column (cols={list(df.columns)}); skipping.")
        return None
    codes = df[c_code].astype(str).str.strip()
    df["soc_code"] = codes.str.slice(0, 7).where(codes.str.match(r"\d{2}-\d{4}"), None)
    score_cols = [c for c in df.columns
                  if re.search(r"expos|score|index|penetrat", str(c), re.I) and c != c_code]
    num = df[score_cols].apply(pd.to_numeric, errors="coerce") if score_cols else None
    if num is None or num.dropna(how="all", axis=1).empty:
        warn("AEI job_exposure: no numeric exposure columns detected; skipping.")
        return None
    df[score_cols] = num
    out = df.dropna(subset=["soc_code"]).groupby("soc_code", as_index=False)[score_cols].mean()
    out = out.rename(columns={c: "aei_" + re.sub(r"\W+", "_", str(c)).strip("_").lower()
                              for c in score_cols})
    log(f"AEI job exposure: {len(out)} occupations")
    return out


# ----------------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------------

def build(pre: str, post: str, skip_aei: bool, skip_optional: bool) -> None:
    pre_df = load_bls_vintage(pre)
    post_df = load_bls_vintage(post)

    pre_b, pre_p = vintage_years(pre)
    post_b, post_p = vintage_years(post)
    pre_pct = f"pct_chg_{pre_b}_{pre_p}"
    post_pct = f"pct_chg_{post_b}_{post_p}"

    panel = post_df.merge(pre_df.drop(columns=["occ_title"], errors="ignore"),
                          on="soc_code", how="outer", indicator="bls_merge")
    panel["bls_merge"] = panel["bls_merge"].map(
        {"both": "both_vintages", "left_only": f"only_{post}", "right_only": f"only_{pre}"})

    
    # The core dependent variable: how much BLS changed its 10-year growth
    # forecast for this occupation between the two vintages, in percentage points.
    panel["revision_pp"] = panel[post_pct] - panel[pre_pct]
    n_both = (panel["bls_merge"] == "both_vintages").sum()
    log(f"panel: {len(panel)} occupations; {n_both} present in both vintages")

    # Record which vintages were used so downstream scripts do not have to
    # hardcode column names.
    panel.attrs["pre_pct_col"] = pre_pct
    panel.attrs["post_pct_col"] = post_pct

    def merge_occ(src: pd.DataFrame | None, name: str) -> None:
        nonlocal panel
        if src is None:
            return
        panel = panel.merge(src, on="soc_code", how="left")
        matched = panel[src.columns.drop("soc_code")[0]].notna().sum()
        log(f"merged {name}: matched {matched}/{len(panel)} occupations")

    if not skip_optional:
        merge_occ(load_felten(), "Felten AIOE")
        merge_occ(load_eloundou_occ(), "Eloundou occupation exposure")
        if not skip_aei:
            merge_occ(load_aei_job_exposure(), "Anthropic Economic Index job exposure")

    # --- task table ----------------------------------------------------------
    tasks = load_onet_tasks()
    if not skip_optional:
        el_t = load_eloundou_tasks()
        if el_t is not None:
            keys = ["onet_code", "task_norm"] if "onet_code" in el_t.columns else ["task_norm"]
            tasks = tasks.merge(el_t, on=keys, how="left")
        if not skip_aei:
            aei_t = load_aei_tasks()
            if aei_t is not None:
                tasks = tasks.merge(aei_t, on="task_norm", how="left")

    # Roll task-level information up to SOC level and attach it to the panel.
    aggs = {"task_id": "count"}
    if "task_importance" in tasks.columns:
        aggs["task_importance"] = "mean"
    if "aei_claude_task_pct" in tasks.columns:
        aggs["aei_claude_task_pct"] = "sum"
    gpts_task_cols = [c for c in tasks.columns if c.startswith("gpts_task_")]
    roll = tasks.groupby("soc_code").agg(aggs).rename(columns={
        "task_id": "n_onet_tasks",
        "task_importance": "mean_task_importance",
        "aei_claude_task_pct": "aei_claude_usage_share",
    }).reset_index()
    for c in gpts_task_cols:  # numeric task labels -> occupation mean; skip the rest
        num = pd.to_numeric(tasks[c], errors="coerce")
        if num.notna().any():
            m = tasks.assign(_v=num).groupby("soc_code")["_v"].mean().rename(c + "_mean")
            roll = roll.merge(m, on="soc_code", how="left")
    panel = panel.merge(roll, on="soc_code", how="left")

    # --- write outputs -------------------------------------------------------
    panel = panel.sort_values("soc_code")
    panel.to_csv(OUT / "occupation_ai_panel.csv", index=False)
    log(f"wrote {OUT / 'occupation_ai_panel.csv'} ({len(panel)} rows x {panel.shape[1]} cols)")

    long = tasks.drop(columns=["task_norm"]).merge(
        panel.drop(columns=["occ_title"], errors="ignore"), on="soc_code", how="left")
    long.to_csv(OUT / "occupation_task_long.csv", index=False)
    log(f"wrote {OUT / 'occupation_task_long.csv'} ({len(long)} rows x {long.shape[1]} cols)")

    # --- sanity summary ------------------------------------------------------
    both = panel[panel["bls_merge"] == "both_vintages"]
    if len(both):
        log(f"sanity check - largest downward growth revisions ({pre} -> {post}), "
            f"percentage points of projected 10-yr growth:")
        cols = ["soc_code", "occ_title", pre_pct, post_pct, "revision_pp"]
        print(both.nsmallest(10, "revision_pp")[cols].to_string(index=False))


def print_manual_list() -> None:
    """List the files to download by hand, for environments that BLS blocks."""
    print("Download these into raw/ using the exact filenames shown.")
    print("Files already present in raw/ are reused and do not need re-downloading.\n")
    for fname, url, desc in MANUAL_SOURCES:
        status = "present" if (RAW / fname).exists() else "MISSING"
        print(f"  [{status:>7}] raw/{fname}")
        print(f"             {desc}")
        print(f"             {url}\n")
    print("Only occupation_*.xlsx (both vintages) and onet_task_statements.xlsx are")
    print("strictly required. The build warns and continues without the others.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", default="2021-31",
                    help="pre-AI BLS vintage (e.g. 2021-31 or 2019-29)")
    ap.add_argument("--post", default=CURRENT_VINTAGE,
                    help="post-AI BLS vintage (e.g. 2024-34 or 2023-33)")
    ap.add_argument("--skip-aei", action="store_true",
                    help="skip Anthropic Economic Index merges")
    ap.add_argument("--skip-optional", action="store_true",
                    help="skip all exposure sources; BLS + O*NET only")
    ap.add_argument("--manual", action="store_true",
                    help="print the files to download by hand into raw/, then exit")
    # Under Jupyter, sys.argv holds the kernel's own arguments (-f kernel.json),
    # which argparse would reject. Ignore them when running inside a notebook.
    argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    a = ap.parse_args(argv)

    RAW.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    if a.manual:
        print_manual_list()
        return

    build(a.pre, a.post, a.skip_aei, a.skip_optional)


if __name__ == "__main__":
    main()

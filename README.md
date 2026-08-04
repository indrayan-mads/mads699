# Did AI expose which occupations are most AI-driven?

The Bureau of Labor Statistics updates and publishes ten-year employment projections for every
occupation in the United States every couple of years. One of those
updates was finalized in September 2022, right before ChatGPT came out. Another was
released in August 2025, a massive gap considering how fast AI moves. 

So there's a natural comparison sitting there: take the same occupation, look at what
the BLS predicted for the occupation before generative AI, opposed to what they predict now. You can see how
much the number has moved, as well as the direction of the move. Another question then arises when we look at 
those occupations. Of those that showed the most change, are those occupations the ones that AI was expected to impact the most
or were they simply the occupations that AI enhanced?

This repo builds the dataset that lets you infer inquiries and define your own answer,
there were four main unsupervised techniques used to derive the results. The four different clustering algorithms 
describe the structure of occupations-space, and runs correlation tests that carry the actual claim to draw valuable insights.

**Short version of the answer:** The general effect looks real if you just correlate AI and occupations
but mostly disappears once you control for the obvious confounds. More on that
below, including why the honest reading may reside somewhere in the middle.

---

![Screenshot](mads699_project_architecture.png)

## How to run it

```bash
pip install -r requirements.txt

python build_ai_jobs_dataset.py        # 1. build the dataset
python analyze_exposure_revision.py    # 2. run the hypothesis tests
python cluster_four_methods.py         # 3. run the four clustering models
python make_figures.py                 # 4. draw the report stats figures
python preprocessing.py                # 5. preprocess for readable figures
python second_prelim_visual.py         # 6. draw the reports figure 1
python third_visual.py                 # 7. draw the reports figure 2
```

Run them in that order. Step 1 writes the panel that steps 2 and 3 read; step 4 reads the
outputs of both. Nothing needs an API key, an account, or a config file.

**It runs offline.** Every input file is committed under `raw/`, so a fresh clone works
with no network access at all. You should not need `--manual`, but it's there if you've
deleted `raw/` and want the download list:

```bash
python build_ai_jobs_dataset.py --manual   # lists every source file + URL, marks what's missing
```

### Useful flags

```bash
python build_ai_jobs_dataset.py --pre 2019-29      # use the older pre-AI vintage
python build_ai_jobs_dataset.py --skip-optional    # BLS + O*NET only, no exposure measures
python analyze_exposure_revision.py --drop-pandemic # drop the 79 COVID-distorted occupations
python cluster_four_methods.py --kmax 6            # search fewer values of k
python cluster_four_methods.py --include-outcome   # the circular specification, for contrast
```

### What's in the repo

```
build_ai_jobs_dataset.py        1. builds the combined dataset
analyze_exposure_revision.py    2. the hypothesis tests and derived variables
cluster_four_methods.py         3. the four unsupervised models
make_figures.py                 4. every figure that appears in the report
requirements.txt
README.md
raw/                            all nine input files, committed
out/                            generated CSVs
figures/                        generated PNGs
```

### Disclaimer 

Many of the scripts in this repo utilize Claude code, specifically the **Opus 4.8 model**, to aid in the programming aspect as well as helped us reduce the typos within our README. 


# JWST exoplanet scraper

This script is designed to do the two-step workflow you asked for:

1. Scrape exoplanet program metadata from the official STScI Cycle GO pages.
2. Use those PIDs to fetch and parse public APT files (with PDF fallback) into an observation-level exoplanet table.

Script created largely thanks to ChatGPT. Program info for Cycles 1-4 was manually extracted by NE. Using this script to fill in Cycle 4 and 5 exoplanets and programs.

## Files

- `jwst_exoplanet_scraper.py`
- `requirements.txt`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run all cycles

```bash
python jwst_exoplanet_scraper.py --cycles 1-5 --outdir output --include-completion
```

This writes:

- `output/jwst_exoplanet_programs.csv`
- `output/jwst_exoplanet_programs.xlsx`
- `output/jwst_exoplanet_observations.csv`
- `output/jwst_exoplanet_observations.xlsx`

## Program-page logic

The scraper looks for these section headings:

- Cycles 1-3:
  - `Exoplanets and Exoplanet Formation`
- Cycles 4-5:
  - `Exoplanet Atmospheres and Habitability`
  - `Exoplanetary System Formation and Dynamics`
  - `Exoplanet System Formation and Dynamics`

That is meant to be resilient to small heading-name changes.

## Observation-level logic

The observation parser intentionally implements the deterministic rules we discussed:

- split by distinct `Observation Number`
- split by explicit target name found in the APT/PDF
- split by optical element (filter / grating / disperser)
- do **not** do domain-expert inference for ambiguous multi-planet direct-imaging systems

So this script should get you the automated, reproducible first pass.
Then you can layer your astrophysical judgment calls on top.

## Important limitations

This is still a heuristic public-file parser. It should work as a robust starting point, but you should expect to manually review:

- ambiguous direct-imaging systems
- cases where APT target names are generic
- programs where the public APT bundle structure differs from the usual pattern
- programs where the public PDF has easier-to-read exposure-time tables than the APT bundle

## Suggested validation workflow

Use your manually curated Cycles 1-3 as validation sets:

1. run Cycle 1
2. compare to your manual CSV
3. adjust any regexes / splitting rules
4. repeat for Cycles 2 and 3
5. once stable, run Cycles 4 and 5

## Example commands

Only part A:

```bash
python jwst_exoplanet_scraper.py --cycles 4,5 --skip-observations --outdir output
```

Part B using an already saved program CSV:

```bash
python jwst_exoplanet_scraper.py --programs-csv output/jwst_exoplanet_programs.csv --skip-programs --outdir output
```

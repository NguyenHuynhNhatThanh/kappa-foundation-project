# Kappa Foundation — BFIN WS2026

## Files
- `Kappa_Foundation_Group_Assignment.ipynb`: notebook split into labeled cells matching the assignment workstreams.
- `Kappa_Foundation_Group_Assignment.py`: sequential Python script export.
- `requirements_kappa.txt`: packages used by the notebook.
- When run, the code creates `data/raw/`, `data/processed/`, and `outputs/`.

## Run in Jupyter
1. Put the notebook, Python script, and requirements file in one working folder.
2. Install packages: `python -m pip install -r requirements_kappa.txt`
3. Start Jupyter: `jupyter lab`
4. Open the notebook and run all cells in order.
5. First run requires internet for Yahoo Finance and Kenneth French downloads.
6. Raw downloads and retrieval metadata are saved under `data/raw/`.
7. To rebuild offline after a successful download, change `DOWNLOAD_DATA = False` in Cell 1 and run all cells again.

## Run as a Python script
`python Kappa_Foundation_Group_Assignment.py`

Edit `DOWNLOAD_DATA` in the script before running offline. Run from the folder where you want `data/` and `outputs/` created.

## Important academic and modelling notes
- The case does not specify inflation, expected returns, risk aversion, or a policy benchmark. The defaults are explicitly labeled working assumptions; the group must justify and freeze them in the initial decision record.
- The stock/ETF universe is a candidate universe, not proof of eligibility as of 31-Dec-2022. Add contemporaneous issuer/exchange evidence and discuss survivorship/selection bias.
- Fictional case bonds are analyzed using supplied cash flows and quotes only; no historical bond prices are fabricated.
- The single-index covariance is a simplified risk model and can understate co-movement from omitted common factors.
- The frozen growth strategy is the sample-covariance optimizer at gamma=5 by default; freeze this choice before evaluating 2023-2025. SPY is the default comparator.
- The inflation goal calculation is a scenario, not a guarantee. Bond coupons are assumed reinvested at the growth sleeve's hypothetical return.
- AI use must follow the course's academic-integrity policy. Disclose material assistance, verify downloads and calculations, and ensure every group member can explain the submitted analysis.

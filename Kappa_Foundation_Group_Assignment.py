"""Kappa Foundation BFIN WS2026 assignment analysis.
Generated from Kappa_Foundation_Group_Assignment.ipynb.
Run top-to-bottom. Requires internet only when DOWNLOAD_DATA=True.
"""


########################################################################################
# Kappa Foundation — BFIN WS2026 Group Assignment
########################################################################################


########################################################################################
# Cell 1 — Setup, assumptions, reproducibility
########################################################################################

from pathlib import Path
import sys, os, json, zipfile, io, re, warnings, platform, subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy
from scipy.optimize import minimize, brentq
import statsmodels.api as sm

SEED = 42
np.random.seed(SEED)

BASE = Path.cwd()
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
OUT = BASE / "outputs"
for folder in (RAW, PROCESSED, OUT):
    folder.mkdir(parents=True, exist_ok=True)

# Case facts: decision date 31-Dec-2022; all amounts in USD.
INITIAL_CASH = 12_000_000
PAYMENTS = {2: 2_400_000, 5: 3_600_000}  # years from 31-Dec-2022
REAL_GOAL_2030 = 8_000_000
INFLATION_BASE = 0.025  # illustrative only; trustees did not specify inflation
EST_START = "2015-01-31"
EST_END = "2022-12-31"
EVAL_START = "2023-01-31"
EVAL_END = "2025-12-31"
DOWNLOAD_DATA = True

# Provisional investment policy assumptions; justify or change before submission.
MAX_WEIGHT = 0.25
RISK_AVERSION_GRID = [2.0, 5.0, 10.0]
CHOSEN_GAMMA = 5.0
PARALLEL_RATE_SHOCK = 0.01
CURVE_SHORT_SHOCK = 0.01
CURVE_LONG_SHOCK = -0.005

VERSION_RECORD = {
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "yfinance": yf.__version__,
    "scipy": scipy.__version__,
    "statsmodels": sm.__version__,
    "download_data": DOWNLOAD_DATA,
    "case_decision_date": "2022-12-31",
    "estimation_window": [EST_START, EST_END],
    "evaluation_window": [EVAL_START, EVAL_END],
    "inflation_assumption": INFLATION_BASE,
}
(OUT / "environment_and_assumptions.json").write_text(json.dumps(VERSION_RECORD, indent=2))
print(json.dumps(VERSION_RECORD, indent=2))


########################################################################################
# Cell 2 — Define the candidate universe and benchmark
########################################################################################

STOCKS = {
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "JPM": "Financials",
    "XOM": "Energy",
    "JNJ": "Health Care",
    "PG": "Consumer Staples",
    "CAT": "Industrials",
    "AMZN": "Consumer Discretionary",
    "NEE": "Utilities",
    "DIS": "Communication Services",
}
ETFS = {
    "SPY": "Broad US large-cap equity benchmark",
    "IWM": "US small-cap equity exposure",
}
TICKERS = list(STOCKS) + list(ETFS)
BENCHMARK = "SPY"
UNIVERSE_RECORD = pd.DataFrame(
    [{"ticker": t, "role": role, "type": "Stock" if t in STOCKS else "ETF"}
     for t, role in {**STOCKS, **ETFS}.items()]
)
UNIVERSE_RECORD.to_csv(OUT / "candidate_universe.csv", index=False)
UNIVERSE_RECORD


########################################################################################
# Cell 3 — Download adjusted market prices from Yahoo Finance
########################################################################################

PRICE_FILE = RAW / "yfinance_daily_adjusted_close.csv"
YF_META_FILE = RAW / "yfinance_download_metadata.json"

if DOWNLOAD_DATA:
    yf_response = yf.download(
        TICKERS, start="2014-12-01", end="2026-01-01",
        interval="1d", auto_adjust=True, actions=False,
        group_by="column", progress=False, threads=True
    )
    if yf_response.empty:
        raise RuntimeError("Yahoo Finance returned no rows. Check internet, ticker eligibility, or retry.")
    # Keep the downloaded response as-is (column names are flattened only for CSV export).
    yf_response.to_csv(PRICE_FILE)
    YF_META_FILE.write_text(json.dumps({
        "source": "Yahoo Finance via yfinance",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": TICKERS, "start_inclusive": "2014-12-01",
        "end_exclusive": "2026-01-01", "interval": "1d",
        "auto_adjust": True, "actions": False,
        "yfinance_version": yf.__version__,
        "note": "CSV stores the downloaded DataFrame; adjusted OHLC fields if returned."
    }, indent=2))
    # Use adjusted Close from the in-memory download.
    if isinstance(yf_response.columns, pd.MultiIndex):
        if "Close" in yf_response.columns.get_level_values(0):
            daily_prices = yf_response["Close"].copy()
        else:
            raise RuntimeError("Expected adjusted Close column missing from Yahoo response.")
    else:
        daily_prices = yf_response[["Close"]].rename(columns={"Close": TICKERS[0]})
else:
    # CSV export from yfinance MultiIndex may have two header rows.
    raw = pd.read_csv(PRICE_FILE, header=[0,1], index_col=0, parse_dates=True)
    if isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
        daily_prices = raw["Close"].copy()
    else:
        raise RuntimeError("Offline reload could not identify Close columns; keep the saved original download.")

daily_prices.index = pd.to_datetime(daily_prices.index)
daily_prices = daily_prices.sort_index()
monthly_prices = daily_prices.resample("ME").last()
monthly_returns = monthly_prices.pct_change(fill_method=None).dropna(how="all")
monthly_prices.to_csv(PROCESSED / "monthly_adjusted_prices.csv")
monthly_returns.to_csv(PROCESSED / "monthly_total_returns.csv")
print("Daily rows:", len(daily_prices), "| monthly return rows:", len(monthly_returns))
print("Missing returns by ticker:\n", monthly_returns.isna().sum())


########################################################################################
# Cell 4 — Retrieve Kenneth French monthly factors
########################################################################################

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

def download_zip(url, destination):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return payload

def parse_french_monthly_csv(zip_bytes, expected_columns):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV found in French ZIP archive.")
        text_csv = zf.read(csv_names[0]).decode("latin1")
    lines = text_csv.splitlines()
    start = None
    for i, line in enumerate(lines):
        first = line.strip().split(",")[0].strip().strip('"')
        if re.fullmatch(r"\d{6}", first):
            start = i
            break
    if start is None:
        raise ValueError("Could not locate YYYYMM data rows in French file.")
    rows = []
    for line in lines[start:]:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if not parts or not re.fullmatch(r"\d{6}", parts[0]):
            break
        if len(parts) >= len(expected_columns) + 1:
            rows.append(parts[:len(expected_columns)+1])
    df = pd.DataFrame(rows, columns=["YYYYMM"] + expected_columns)
    df["Date"] = pd.to_datetime(df["YYYYMM"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("Date").drop(columns="YYYYMM")
    for col in expected_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    return df

if DOWNLOAD_DATA:
    ff5_bytes = download_zip(FF5_URL, RAW / "F-F_Research_Data_5_Factors_2x3_CSV.zip")
    mom_bytes = download_zip(MOM_URL, RAW / "F-F_Momentum_Factor_CSV.zip")
    (RAW / "french_download_metadata.json").write_text(json.dumps({
        "source": "Kenneth French Data Library",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"url": FF5_URL, "saved_as": "F-F_Research_Data_5_Factors_2x3_CSV.zip"},
            {"url": MOM_URL, "saved_as": "F-F_Momentum_Factor_CSV.zip"}
        ],
        "units_in_download": "percent per month; converted to decimal in processed data"
    }, indent=2))
else:
    ff5_bytes = (RAW / "F-F_Research_Data_5_Factors_2x3_CSV.zip").read_bytes()
    mom_bytes = (RAW / "F-F_Momentum_Factor_CSV.zip").read_bytes()

ff5 = parse_french_monthly_csv(ff5_bytes, ["Mkt-RF","SMB","HML","RMW","CMA","RF"])
mom = parse_french_monthly_csv(mom_bytes, ["Mom"])
factors = ff5.join(mom, how="inner").sort_index()
factors.to_csv(PROCESSED / "kenneth_french_monthly_factors_decimal.csv")
print(factors.head())
print("Factor coverage:", factors.index.min().date(), "to", factors.index.max().date())


########################################################################################
# Cell 5 — Validate the common estimation/evaluation samples
########################################################################################

est_start = pd.Timestamp("2015-01-01")
est_end = pd.Timestamp("2022-12-31")
eval_start = pd.Timestamp("2023-01-01")
eval_end = pd.Timestamp("2025-12-31")

asset_returns = monthly_returns.loc[:, TICKERS].copy()
common = asset_returns.join(factors, how="inner")
est = common.loc[(common.index >= est_start) & (common.index <= est_end)].copy()
evaluation = common.loc[(common.index >= eval_start) & (common.index <= eval_end)].copy()

# Keep a strict common sample for fair covariance/model comparisons.
est = est.dropna(subset=TICKERS + ["Mkt-RF","RF","SMB","HML","RMW","CMA","Mom"])
evaluation = evaluation.dropna(subset=TICKERS + ["Mkt-RF","RF","SMB","HML","RMW","CMA","Mom"])

if len(est) < 60:
    raise ValueError(f"Only {len(est)} common monthly observations in estimation window; need at least 60.")
if len(evaluation) < 30:
    warnings.warn(f"Evaluation window has {len(evaluation)} common months; expected about 36.")

coverage = pd.DataFrame({
    "period": ["Estimation", "Evaluation"],
    "start": [est.index.min().date(), evaluation.index.min().date()],
    "end": [est.index.max().date(), evaluation.index.max().date()],
    "common_months": [len(est), len(evaluation)]
})
coverage.to_csv(OUT / "sample_coverage.csv", index=False)
coverage


########################################################################################
# Cell 6 — Bond valuation, YTM, duration and convexity
########################################################################################

BONDS = pd.DataFrame([
    {"bond":"Alpha", "maturity":1, "coupon":0.0000, "price":958.77},
    {"bond":"Beta", "maturity":2, "coupon":0.0000, "price":922.78},
    {"bond":"Gamma", "maturity":3, "coupon":0.0400, "price":1002.49},
    {"bond":"Delta", "maturity":5, "coupon":0.0350, "price":986.07},
    {"bond":"Epsilon", "maturity":8, "coupon":0.0450, "price":1040.71},
])
FACE = 1000.0

def bond_cashflows(face, coupon_rate, maturity):
    times = np.arange(1, maturity + 1, dtype=float)
    cf = np.full(maturity, face * coupon_rate)
    cf[-1] += face
    return times, cf

def price_from_yield(y, face, coupon_rate, maturity):
    t, cf = bond_cashflows(face, coupon_rate, maturity)
    return float(np.sum(cf / (1+y)**t))

def ytm_from_price(price, face, coupon_rate, maturity):
    return brentq(lambda y: price_from_yield(y, face, coupon_rate, maturity)-price, -0.95, 2.0)

def bond_metrics(row):
    y = ytm_from_price(row.price, FACE, row.coupon, int(row.maturity))
    t, cf = bond_cashflows(FACE, row.coupon, int(row.maturity))
    pv = cf / (1+y)**t
    macaulay = np.sum(t * pv) / row.price
    mod_duration = macaulay / (1+y)
    convexity = np.sum(cf * t * (t+1) / (1+y)**(t+2)) / row.price
    up = price_from_yield(y + PARALLEL_RATE_SHOCK, FACE, row.coupon, int(row.maturity))
    down = price_from_yield(y - PARALLEL_RATE_SHOCK, FACE, row.coupon, int(row.maturity))
    duration_approx_up = row.price * (-mod_duration*PARALLEL_RATE_SHOCK + 0.5*convexity*PARALLEL_RATE_SHOCK**2)
    duration_approx_down = row.price * (mod_duration*PARALLEL_RATE_SHOCK + 0.5*convexity*PARALLEL_RATE_SHOCK**2)
    return pd.Series({
        "ytm": y, "macaulay_duration": macaulay, "modified_duration": mod_duration,
        "convexity": convexity, "price_if_yield_plus_100bp": up,
        "exact_change_plus_100bp": up/row.price-1,
        "duration_convexity_change_plus_100bp": duration_approx_up/row.price,
        "price_if_yield_minus_100bp": down,
        "exact_change_minus_100bp": down/row.price-1,
        "duration_convexity_change_minus_100bp": duration_approx_down/row.price
    })

bond_results = pd.concat([BONDS, BONDS.apply(bond_metrics, axis=1)], axis=1)
bond_results.to_csv(OUT / "bond_valuation_duration_convexity.csv", index=False)
bond_results


########################################################################################
# Cell 7 — Fund the mandatory payments with case bonds
########################################################################################

beta_face = PAYMENTS[2]  # zero-coupon face amount matures at $2.4m
delta_face = PAYMENTS[5]  # principal alone meets $3.6m; coupon is extra
beta_cost = beta_face / FACE * float(BONDS.loc[BONDS.bond.eq("Beta"), "price"].iloc[0])
delta_cost = delta_face / FACE * float(BONDS.loc[BONDS.bond.eq("Delta"), "price"].iloc[0])
bond_cost = beta_cost + delta_cost
growth_capital = INITIAL_CASH - bond_cost

funding_summary = pd.DataFrame([
    {"asset":"Beta zero-coupon", "face_or_units":beta_face, "initial_cost":beta_cost,
     "contractual_maturity_cashflow":beta_face, "payment_obligation":PAYMENTS[2], "surplus_at_maturity":0.0},
    {"asset":"Delta 3.5% coupon", "face_or_units":delta_face, "initial_cost":delta_cost,
     "contractual_maturity_cashflow":delta_face*(1+BONDS.loc[BONDS.bond.eq("Delta"),"coupon"].iloc[0]),
     "payment_obligation":PAYMENTS[5], "surplus_at_maturity":delta_face*BONDS.loc[BONDS.bond.eq("Delta"),"coupon"].iloc[0]},
    {"asset":"Growth allocation", "face_or_units":np.nan, "initial_cost":growth_capital,
     "contractual_maturity_cashflow":np.nan, "payment_obligation":0.0, "surplus_at_maturity":np.nan},
])
funding_summary.to_csv(OUT / "initial_funding_plan.csv", index=False)
print(f"Beta cost: ${beta_cost:,.2f}")
print(f"Delta cost: ${delta_cost:,.2f}")
print(f"Remaining for growth allocation: ${growth_capital:,.2f}")
print(f"Check total: ${bond_cost + growth_capital:,.2f}")
funding_summary


########################################################################################
# Cell 8 — Incomplete term structure and curve-shape stress
########################################################################################

zero_quotes = BONDS[BONDS.bond.isin(["Alpha","Beta"])].copy()
zero_quotes["spot_rate"] = (FACE / zero_quotes["price"])**(1/zero_quotes["maturity"]) - 1
r1 = float(zero_quotes.loc[zero_quotes.maturity.eq(1),"spot_rate"].iloc[0])
r2 = float(zero_quotes.loc[zero_quotes.maturity.eq(2),"spot_rate"].iloc[0])
base_spot = {1:r1, 2:r2}
for mat in range(3, 9):
    base_spot[mat] = r2  # explicit flat extension beyond observed zero quotes

def price_from_spot_curve(coupon_rate, maturity, face=1000.0, curve=None):
    times, cf = bond_cashflows(face, coupon_rate, maturity)
    return float(sum(cf_i/(1+curve[int(t)])**t for t, cf_i in zip(times, cf)))

def shock_curve(curve, short_shock=0.01, long_shock=-0.005):
    # Linear shock profile: +100bp at 1y, gradually reaching -50bp at 8y.
    out = {}
    for t, rate in curve.items():
        weight = (t-1)/(8-1)
        out[t] = rate + short_shock*(1-weight) + long_shock*weight
    return out

curve_up = {t:r+PARALLEL_RATE_SHOCK for t,r in base_spot.items()}
curve_twist = shock_curve(base_spot, CURVE_SHORT_SHOCK, CURVE_LONG_SHOCK)
curve_rows = []
for _, row in BONDS.iterrows():
    base_p = price_from_spot_curve(row.coupon, int(row.maturity), curve=base_spot)
    parallel_p = price_from_spot_curve(row.coupon, int(row.maturity), curve=curve_up)
    twist_p = price_from_spot_curve(row.coupon, int(row.maturity), curve=curve_twist)
    curve_rows.append({"bond":row.bond, "quoted_price":row.price, "scenario_base_price":base_p,
                       "parallel_up_100bp_price":parallel_p, "parallel_up_change":parallel_p/base_p-1,
                       "curve_twist_price":twist_p, "curve_twist_change":twist_p/base_p-1})
curve_stress = pd.DataFrame(curve_rows)
curve_stress.to_csv(OUT / "bond_curve_shape_stress.csv", index=False)
print("Observed zero rates:", {"1y":r1, "2y":r2})
curve_stress


########################################################################################
# Cell 9 — Real-value goal and required growth return
########################################################################################

def terminal_value_given_return(r, growth_capital, delta_face, coupon_rate=0.035):
    # Growth capital compounds 8 years; Delta pays annual coupons in years 1-5.
    coupons = delta_face * coupon_rate
    return growth_capital*(1+r)**8 + sum(coupons*(1+r)**(8-t) for t in range(1,6))

def required_return_for_inflation(inflation):
    target = REAL_GOAL_2030*(1+inflation)**8
    f = lambda r: terminal_value_given_return(r, growth_capital, delta_face)-target
    try:
        req = brentq(f, -0.90, 1.0)
    except ValueError:
        req = np.nan
    return {"annual_inflation":inflation, "nominal_2030_target":target,
            "required_growth_return":req}

inflation_sensitivity = pd.DataFrame([required_return_for_inflation(i) for i in [0.00,0.02,0.025,0.03,0.04]])
inflation_sensitivity.to_csv(OUT / "inflation_goal_sensitivity.csv", index=False)
inflation_sensitivity


########################################################################################
# Cell 10 — Risk–return preference comparison
########################################################################################

R_EST = est[TICKERS].copy()
mu_ann = R_EST.mean() * 12
cov_sample_ann = R_EST.cov() * 12

def utility_table(mu, cov, gamma_values):
    rows=[]
    for ticker in mu.index:
        vol = np.sqrt(max(cov.loc[ticker,ticker],0))
        for gamma in gamma_values:
            utility = mu[ticker] - 0.5*gamma*cov.loc[ticker,ticker]
            rows.append({"asset":ticker,"gamma":gamma,"expected_return_ann":mu[ticker],
                         "volatility_ann":vol,"variance_ann":cov.loc[ticker,ticker],"utility":utility})
    return pd.DataFrame(rows)

risk_pref = utility_table(mu_ann, cov_sample_ann, RISK_AVERSION_GRID)
risk_pref.to_csv(OUT / "risk_return_preference_comparison.csv", index=False)
risk_pref.sort_values(["gamma","utility"], ascending=[True,False]).groupby("gamma").head(5)


########################################################################################
# Cell 11 — Build sample-covariance and single-index risk estimates
########################################################################################

# Estimate market model using SPY monthly total return as the market proxy.
market = R_EST[BENCHMARK]
betas, residual_vars = {}, {}
for ticker in TICKERS:
    X = sm.add_constant(market)
    fit = sm.OLS(R_EST[ticker], X, missing="drop").fit()
    betas[ticker] = float(fit.params[market.name])
    residual_vars[ticker] = float(np.var(fit.resid, ddof=1))
market_var = float(np.var(market, ddof=1))
cov_single_month = pd.DataFrame(
    [[betas[i]*betas[j]*market_var + (residual_vars[i] if i==j else 0.0)
      for j in TICKERS] for i in TICKERS],
    index=TICKERS, columns=TICKERS
)
cov_single_ann = cov_single_month * 12
cov_sample_ann = cov_sample_ann.loc[TICKERS, TICKERS]
mu_ann = mu_ann.loc[TICKERS]

def optimize_weights(mu, cov, gamma, max_weight=MAX_WEIGHT):
    n = len(mu)
    x0 = np.repeat(1/n, n)
    bounds = [(0.0, max_weight) for _ in range(n)]
    objective = lambda w: -(float(np.dot(w,mu)) - 0.5*gamma*float(w @ cov @ w))
    result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                      constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                      options={"maxiter":2000,"ftol":1e-12})
    if not result.success:
        warnings.warn("Optimization warning: " + result.message)
    w = np.clip(result.x, 0, max_weight)
    w = w / w.sum()
    return pd.Series(w, index=mu.index, name=f"gamma_{gamma:g}")

optimization_rows=[]
weights_store={}
for gamma in RISK_AVERSION_GRID:
    for risk_name, cov in [("Sample covariance",cov_sample_ann),("Single-index",cov_single_ann)]:
        w = optimize_weights(mu_ann, cov, gamma)
        weights_store[(risk_name,gamma)] = w
        optimization_rows.extend([{"risk_model":risk_name,"gamma":gamma,"ticker":t,"weight":w[t],
                                   "expected_return_ann":float(w @ mu_ann),
                                   "volatility_ann":float(np.sqrt(w @ cov @ w))}
                                  for t in TICKERS])
optimization_results = pd.DataFrame(optimization_rows)
optimization_results.to_csv(OUT / "portfolio_optimization_weights.csv", index=False)
optimization_results.head(12)


########################################################################################
# Cell 12 — Freeze the initial growth strategy and benchmark
########################################################################################

FROZEN_STRATEGY = weights_store[("Sample covariance", CHOSEN_GAMMA)].copy()
FROZEN_BENCHMARK = BENCHMARK
FROZEN_STRATEGY.rename("weight").to_csv(OUT / "frozen_growth_strategy_weights.csv")
print("Frozen strategy weights (must be dated/frozen in initial decision record):")
print(FROZEN_STRATEGY[FROZEN_STRATEGY > 1e-6].sort_values(ascending=False).to_string())
print("Frozen benchmark:", FROZEN_BENCHMARK)


########################################################################################
# Cell 13 — Market-only and Fama–French multifactor regressions
########################################################################################

def portfolio_return(returns, weights):
    return returns[weights.index].mul(weights, axis=1).sum(axis=1)

def fit_factor_models(asset_ret, factor_frame):
    joined = pd.concat([asset_ret.rename("asset"), factor_frame], axis=1).dropna()
    y = joined["asset"] - joined["RF"]
    market_X = sm.add_constant(joined[["Mkt-RF"]])
    ff6_X = sm.add_constant(joined[["Mkt-RF","SMB","HML","RMW","CMA","Mom"]])
    mkt = sm.OLS(y, market_X).fit()
    ff6 = sm.OLS(y, ff6_X).fit()
    def summarize(model, name):
        return {"model":name, "alpha_monthly":model.params["const"],
                "alpha_annual_approx":model.params["const"]*12,
                "alpha_pvalue":model.pvalues["const"], "r_squared":model.rsquared,
                "observations":int(model.nobs),
                "coefficients":json.dumps({k:float(v) for k,v in model.params.items()})}
    return [summarize(mkt,"Market-only"), summarize(ff6,"FF5 + Momentum")]

growth_est = portfolio_return(est, FROZEN_STRATEGY)
regression_rows = []
for name, series in [("Frozen growth portfolio",growth_est),
                     ("AAPL",est["AAPL"]),("JPM",est["JPM"])]:
    for row in fit_factor_models(series, est[["Mkt-RF","SMB","HML","RMW","CMA","Mom","RF"]]):
        row["asset"] = name
        regression_rows.append(row)
factor_results = pd.DataFrame(regression_rows)
factor_results.to_csv(OUT / "factor_regression_results.csv", index=False)
factor_results[["asset","model","alpha_annual_approx","alpha_pvalue","r_squared","observations"]]


########################################################################################
# Cell 14 — Frozen 2023–2025 evaluation
########################################################################################

def max_drawdown(r):
    # Include initial wealth of 1.0 so a loss in the first month counts as drawdown.
    wealth = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.offsets.MonthEnd(1)]),
                        (1+r).cumprod()])
    peak = wealth.cummax()
    return float((wealth/peak-1).min())

def performance_stats(r, rf=None):
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {}
    cumulative = float((1+r).prod()-1)
    cagr = float((1+cumulative)**(12/n)-1)
    vol = float(r.std(ddof=1)*np.sqrt(12))
    rf = pd.Series(0.0, index=r.index) if rf is None else rf.reindex(r.index).fillna(0.0)
    excess = r-rf
    sharpe = float(excess.mean()/excess.std(ddof=1)*np.sqrt(12)) if excess.std(ddof=1)>0 else np.nan
    return {"months":n,"cumulative_return":cumulative,"CAGR":cagr,
            "annualized_volatility":vol,"max_drawdown":max_drawdown(r),"Sharpe":sharpe}

eval_returns = monthly_returns.loc[(monthly_returns.index>=eval_start)&(monthly_returns.index<=eval_end), TICKERS]
eval_returns = eval_returns.dropna(subset=TICKERS)
frozen_growth_eval = portfolio_return(eval_returns, FROZEN_STRATEGY)
benchmark_eval = eval_returns[FROZEN_BENCHMARK]
rf_eval = factors.loc[factors.index.isin(eval_returns.index), "RF"]
performance = pd.DataFrame([
    {"strategy":"Frozen growth strategy", **performance_stats(frozen_growth_eval, rf_eval)},
    {"strategy":f"Frozen benchmark ({FROZEN_BENCHMARK})", **performance_stats(benchmark_eval, rf_eval)}
])
performance.to_csv(OUT / "frozen_strategy_vs_benchmark_2023_2025.csv", index=False)
performance


########################################################################################
# Cell 15 — Sensitivity: find a threshold that can reverse the decision
########################################################################################

sensitivity_rows=[]
for gamma in np.arange(0.5, 15.01, 0.5):
    for risk_name, cov in [("Sample covariance",cov_sample_ann),("Single-index",cov_single_ann)]:
        w=optimize_weights(mu_ann,cov,float(gamma))
        sensitivity_rows.append({
            "gamma":float(gamma), "risk_model":risk_name,
            "expected_return_ann":float(w @ mu_ann),
            "volatility_ann":float(np.sqrt(w @ cov @ w)),
            "largest_weight":float(w.max()),
            "largest_holding":str(w.idxmax()),
            "weight_SPY":float(w.get("SPY",0)),
            "weight_IWM":float(w.get("IWM",0)),
            "number_of_positions_over_5pct":int((w>0.05).sum())
        })
gamma_sensitivity=pd.DataFrame(sensitivity_rows)
gamma_sensitivity.to_csv(OUT / "risk_aversion_threshold_sensitivity.csv", index=False)
gamma_sensitivity.head(10)


########################################################################################
# Cell 16 — Decision log and monitoring framework
########################################################################################

decision_log = pd.DataFrame([
    {"decision":"Inflation assumption","initial_choice":str(INFLATION_BASE),
     "reason":"Illustrative baseline; case leaves inflation unspecified",
     "evidence_that_would_change_it":"Trustee mandate or defensible external inflation scenario"},
    {"decision":"Growth benchmark","initial_choice":FROZEN_BENCHMARK,
     "reason":"Simple, investable, transparent broad US equity comparator",
     "evidence_that_would_change_it":"Mandate specifies a different investable policy benchmark"},
    {"decision":"Risk aversion","initial_choice":str(CHOSEN_GAMMA),
     "reason":"Working value to be compared with lower/higher gamma",
     "evidence_that_would_change_it":"Trustee agreement on loss metric, horizon, and acceptable trade-off"},
    {"decision":"Evaluation rule","initial_choice":"Frozen weights; monthly total returns; Jan-2023 to Dec-2025",
     "reason":"Prevents hindsight re-optimization",
     "evidence_that_would_change_it":"Only a formally revised mandate; retain original results"}
])
decision_log.to_csv(OUT / "decision_log_template.csv", index=False)
decision_log


########################################################################################
# Cell 17 — Export final tables and charts
########################################################################################

# Comparison plot: frozen growth sleeve vs benchmark wealth indices
wealth = pd.concat([
    (1+frozen_growth_eval).cumprod().rename("Frozen growth strategy"),
    (1+benchmark_eval).cumprod().rename(f"Benchmark ({FROZEN_BENCHMARK})")
], axis=1)
ax = wealth.plot(title="Frozen growth strategy vs benchmark (Jan 2023–Dec 2025)",
                 ylabel="Growth of $1", xlabel="Month", figsize=(10,5))
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(OUT / "frozen_strategy_vs_benchmark.png", dpi=160)
plt.show()

print("Outputs saved to:", OUT.resolve())
print("\nFiles:")
for p in sorted(OUT.iterdir()):
    print("-", p.name)

"""Recorded hyperparameter searches for a gradient-boosted tree classifier on Spambase.

Used by src/figures/cs-classifier-many.ts (chapter "Tuning a Classifier").

Data: Spambase (Hopkins, Reeber, Forman, and Suermondt, 1999), UCI Machine
Learning Repository, https://doi.org/10.24432/C53G6X, licensed CC BY 4.0;
fetched from OpenML (data id 44). 4,601 e-mails, 57 numeric features, 39.4%
spam. scikit-learn is BSD-3-Clause.

Model: sklearn.ensemble.HistGradientBoostingClassifier (early stopping off,
random_state 0) with seven hyperparameters, each mapped from [0, 1]:

    learning_rate      0.005 .. 1      log
    max_iter           10 .. 500       log, integer   (number of trees)
    max_leaf_nodes     2 .. 256        log, integer
    min_samples_leaf   1 .. 200        log, integer
    l2_regularization  1e-4 .. 10      log
    max_features       0.1 .. 1        linear         (fraction of features per split)
    max_bins           4 .. 255        log, integer

Protocol: a stratified 75/25 split (random_state 0) into a tuning set (3,450
e-mails) and a held-out test set (1,151). The objective is the 5-fold
stratified cross-validation error on the tuning set (one fixed split,
random_state 0). For every evaluated configuration we also record the
fold-to-fold standard deviation, the error on the held-out test set of the
model refitted on the whole tuning set (never seen by the optimizers), and the
wall-clock seconds of the cross-validation (one CPU thread, Apple M-series).

Optimizers, ten seeds each, 60 evaluations per run:
  random  uniform in the unit cube.
  bo      the first 10 points of the random run with the same seed, then
          Gaussian-process expected improvement: a Matern 5/2 kernel with one
          lengthscale per hyperparameter (ARD), a constant amplitude and a
          white-noise term, all fitted by maximum marginal likelihood
          (scikit-learn GaussianProcessRegressor, normalize_y, 4 restarts), on
          log(CV error); EI against the lowest posterior mean at an evaluated
          point; maximized over 4,000 uniform candidates plus 2,000 Gaussian
          perturbations (sd 0.1) of the five best points.

Reproduce (about 10 minutes on 16 cores):

    uv venv .cache/venv-cases --python 3.12
    uv pip install --python .cache/venv-cases/bin/python numpy scipy scikit-learn pandas
    .cache/venv-cases/bin/python tools/figure-data/cs-classifier-gbm.py

Writes src/figures/data/cs-classifier-gbm.json and prints the summaries quoted
in the chapter; with --summary it only prints the summaries from the JSON. Versions: Python 3.12, scikit-learn 1.9.1 (2026-10-02).
"""

import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(ROOT, "src", "figures", "data", "cs-classifier-gbm.json")
DATA_HOME = os.path.join(ROOT, ".cache", "sk-data")

SEEDS = list(range(1, 11))
BUDGET = 60
N_INIT = 10

# name, low, high, scale, integer
DIMS = [
    ("learning_rate", 0.005, 1.0, "log", False),
    ("max_iter", 10, 500, "log", True),
    ("max_leaf_nodes", 2, 256, "log", True),
    ("min_samples_leaf", 1, 200, "log", True),
    ("l2_regularization", 1e-4, 10.0, "log", False),
    ("max_features", 0.1, 1.0, "linear", False),
    ("max_bins", 4, 255, "log", True),
]
D = len(DIMS)


def decode(u):
    cfg = {}
    for v, (name, lo, hi, scale, integer) in zip(u, DIMS):
        v = min(1.0, max(0.0, float(v)))
        x = math.exp(math.log(lo) + v * (math.log(hi) - math.log(lo))) if scale == "log" else lo + v * (hi - lo)
        cfg[name] = int(round(x)) if integer else x
    return cfg


def encode(cfg):
    u = []
    for name, lo, hi, scale, _ in DIMS:
        x = cfg[name]
        u.append((math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)) if scale == "log" else (x - lo) / (hi - lo))
    return u


def load():
    d = fetch_openml(data_id=44, as_frame=True, data_home=DATA_HOME)
    X = d.data.to_numpy(float)
    y = d.target.astype(int).to_numpy()
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


XT, XH, YT, YH = load()
CV = list(StratifiedKFold(5, shuffle=True, random_state=0).split(XT, YT))


def model(cfg):
    return HistGradientBoostingClassifier(early_stopping=False, random_state=0, **cfg)


def evaluate(cfg):
    t0 = time.perf_counter()
    errs = []
    for tr, va in CV:
        m = model(cfg).fit(XT[tr], YT[tr])
        errs.append(float((m.predict(XT[va]) != YT[va]).mean()))
    secs = time.perf_counter() - t0
    m = model(cfg).fit(XT, YT)
    test = float((m.predict(XH) != YH).mean())
    return float(np.mean(errs)), float(np.std(errs, ddof=1)), test, secs


def random_points(seed, n):
    return np.random.default_rng(seed).random((n, D))


def propose(U, y, rng):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        k = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.full(D, 0.5), length_scale_bounds=(0.03, 30.0), nu=2.5) \
            + WhiteKernel(1e-2, (1e-6, 1.0))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=4, random_state=int(rng.integers(1 << 30)))
        gp.fit(U, y)
    best = float(gp.predict(U).min())  # minimizing log error
    top = U[np.argsort(y)[:5]]
    cand = np.vstack([rng.random((4000, D)), np.clip(top[rng.integers(0, len(top), 2000)] + 0.1 * rng.standard_normal((2000, D)), 0, 1)])
    mu, sd = gp.predict(cand, return_std=True)
    sd = np.maximum(sd, 1e-9)
    z = (best - mu) / sd
    ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
    return cand[int(np.argmax(ei))], gp


def run(job):
    method, seed = job
    rows = []
    init = random_points(seed, BUDGET)
    rng = np.random.default_rng(1000 + seed)
    U, y = [], []
    for n in range(BUDGET):
        if method == "random" or n < N_INIT:
            u = init[n]
        else:
            u, _ = propose(np.array(U), np.array(y), rng)
        cfg = decode(u)
        uq = encode(cfg)  # the configuration actually evaluated, after rounding
        cv, sd, test, secs = evaluate(cfg)
        U.append(uq)
        y.append(math.log(cv))
        rows.append([round(v, 4) for v in uq] + [round(cv, 5), round(sd, 5), round(test, 5), round(secs, 3)])
        print(f"{method} {seed} {n} cv={cv:.4f} test={test:.4f} {secs:.1f}s", file=sys.stderr, flush=True)
    return {"method": method, "seed": seed, "evals": rows}


def importance(runs):
    """Main-effect share of each hyperparameter from the random-search evaluations.

    Random search samples the unit cube uniformly, so binning one coordinate
    into quintiles and computing the between-bin share of the variance of
    log(CV error) estimates that coordinate's first-order (main-effect) share.
    """
    R = np.array([e for r in runs if r["method"] == "random" for e in r["evals"]])
    U, y = R[:, :D], np.log(R[:, D])
    out = {}
    for k, (name, *_rest) in enumerate(DIMS):
        bins = np.minimum((U[:, k] * 5).astype(int), 4)
        between = sum((bins == b).sum() * (y[bins == b].mean() - y.mean()) ** 2 for b in range(5) if (bins == b).any())
        out[name] = between / ((y - y.mean()) ** 2).sum()
    return out


def main():
    jobs = [(m, s) for s in SEEDS for m in ("random", "bo")]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        runs = list(ex.map(run, jobs))
    default_cfg = {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                   "l2_regularization": 0.0, "max_features": 1.0, "max_bins": 255}
    dcv, dsd, dtest, dsecs = evaluate(default_cfg)
    data = {
        "about": "HistGradientBoostingClassifier on Spambase (UCI, CC BY 4.0): recorded random-search and GP-EI runs; see tools/figure-data/cs-classifier-gbm.py",
        "nTune": int(len(YT)), "nTest": int(len(YH)),
        "dims": [{"name": n, "lo": lo, "hi": hi, "scale": s, "int": i} for n, lo, hi, s, i in DIMS],
        # each eval: u_1..u_7 (unit-cube coordinates), CV error, fold sd, test error, seconds
        "default": {"cv": round(dcv, 5), "sd": round(dsd, 5), "test": round(dtest, 5), "secs": round(dsecs, 3)},
        "budget": BUDGET, "init": N_INIT,
        "runs": runs,
        "sklearn": sklearn.__version__,
    }
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
    summarize(data)


def summarize(data):
    """Every number about these runs that the chapter quotes."""
    runs = data["runs"]
    dflt = data["default"]
    print(f"sklearn default: cv {dflt['cv']:.4f} (fold sd {dflt['sd']:.4f}), test {dflt['test']:.4f}, {dflt['secs']:.1f}s")
    for method in ("random", "bo"):
        rs = [r for r in runs if r["method"] == method]
        for n in (10, 20, 30, 60):
            inc = np.array([min(r["evals"][:n], key=lambda e: e[D])[D:D + 3:2] for r in rs])
            print(f"{method} after {n}: best CV median {np.median(inc[:, 0]):.4f} [{inc[:, 0].min():.4f}, {inc[:, 0].max():.4f}], "
                  f"its test median {np.median(inc[:, 1]):.4f} [{inc[:, 1].min():.4f}, {inc[:, 1].max():.4f}]")
        curves = np.array([np.minimum.accumulate([e[D] for e in r["evals"]]) for r in rs])
        print(f"{method}: median best-so-far CV at 10..60: {np.round(100 * np.median(curves, 0)[9::10], 2)}")
        secs = np.array([e[D + 3] for r in rs for e in r["evals"]])
        print(f"{method}: seconds per run median {np.median([sum(e[D + 3] for e in r['evals']) for r in rs]):.0f}, per eval median {np.median(secs):.2f}")
        for budget_s in (92,):
            best = []
            for r in rs:
                cum, b = 0.0, None
                for e in r["evals"]:
                    cum += e[D + 3]
                    if cum > budget_s:
                        break
                    b = e[D] if b is None else min(b, e[D])
                best.append(b)
            print(f"{method}: best CV within the first {budget_s} s of compute, median {np.median(best):.4f}")
        U = np.array([e[:D] for r in rs for e in r["evals"][data["init"]:]])
        edge = (U < 0.02) | (U > 0.98)
        print(f"{method}: evaluations after the first {data['init']} with a coordinate within 2% of an edge: {edge.any(1).mean():.2f}")
        F = np.array([min(r["evals"], key=lambda e: e[D])[:D] for r in rs])
        print(f"{method}: final picks with a coordinate within 2% of an edge: {((F < 0.02) | (F > 0.98)).any(1).sum()} of {len(rs)}")
    allev = [e for r in runs for e in r["evals"]]
    secs = np.array([e[D + 3] for e in allev])
    print(f"seconds per evaluation: min {secs.min():.2f}, median {np.median(secs):.2f}, max {secs.max():.1f}")
    cv = np.array([e[D] for e in allev])
    test = np.array([e[D + 2] for e in allev])
    good = cv < 0.05
    print(f"all evaluations: CV min {cv.min():.4f}, median {np.median(cv):.4f}, max {cv.max():.4f}; 10% quantile {np.quantile(cv, 0.1):.5f}")
    print(f"CV-test correlation: all {np.corrcoef(cv, test)[0, 1]:.2f}; among CV < 5% ({good.sum()} configurations) {np.corrcoef(cv[good], test[good])[0, 1]:.2f}")
    print(f"fold sd among CV < 5%: median {np.median([e[D + 1] for e in allev if e[D] < 0.05]):.4f}")
    rcv = np.array([e[D] for r in runs if r["method"] == "random" for e in r["evals"]])
    print(f"random configurations beating the default's CV: {(rcv < dflt['cv']).sum()} of {len(rcv)}")
    nt = data["nTest"]
    print(f"binomial standard error of a test error near {dflt['test']:.3f} on {nt} e-mails: {math.sqrt(dflt['test'] * (1 - dflt['test']) / nt):.4f}")
    lr = np.array([decode(e[:D])["learning_rate"] * decode(e[:D])["max_iter"] for e in allev])
    top = cv <= np.quantile(cv, 0.1)
    print(f"learning_rate x max_iter: median {np.median(lr):.1f} overall, {np.median(lr[top]):.1f} in the best 10%")
    shares = importance(runs)
    for name, v in sorted(shares.items(), key=lambda kv: -kv[1]):
        print(f"main-effect share {name}: {v:.3f}")
    print(f"sum of main-effect shares: {sum(shares.values()):.2f}")


if "--summary" in sys.argv:
    summarize(json.load(open(OUT)))
    sys.exit(0)


if __name__ == "__main__":
    main()

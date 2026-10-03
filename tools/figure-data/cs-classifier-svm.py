"""Measured validation-error landscape of an RBF-kernel SVM on handwritten digits.

Used by src/figures/cs-classifier-landscape.ts (chapter "Tuning a Classifier").

Data: scikit-learn's bundled `load_digits`, 1,797 8x8 images of handwritten
digits 0 to 9. It is a copy of the test set of the UCI "Optical Recognition of
Handwritten Digits" data set (Alpaydin and Kaynak, 1998;
https://doi.org/10.24432/C50P49), licensed CC BY 4.0 by the UCI Machine
Learning Repository. scikit-learn is BSD-3-Clause.

Model: MinMaxScaler to [0, 1] (fitted inside each training fold), then
sklearn.svm.SVC with an RBF kernel. Hyperparameters on the ranges recommended
by Hsu, Chang, and Lin, "A Practical Guide to Support Vector Classification"
(the coarse grid C = 2^-5..2^15, gamma = 2^-15..2^3), sampled every half
power of two: 41 values of log2 C and 37 values of log2 gamma, 1,517 settings.

Measurement: each setting is evaluated by stratified 5-fold cross-validation,
repeated with R = 5 different random splits (seeds 0..4). For each setting and
split we store the total number of misclassified images over the five folds
(out of 1,797), so the validation error of one evaluation is count / 1797. The
five repeats are five real measurements of the same setting; the figure draws
one of them per evaluation, so its "noise" is measured, not simulated. The
fold-to-fold spread is summarized on stdout.

Reproduce (about 3 minutes on 16 cores):

    uv venv .cache/venv-cases --python 3.12
    uv pip install --python .cache/venv-cases/bin/python numpy scipy scikit-learn
    .cache/venv-cases/bin/python tools/figure-data/cs-classifier-svm.py

Writes src/figures/data/cs-classifier-svm.json and prints the summaries quoted
in the chapter. With --summary it skips the measurement and prints the
summaries from the existing JSON. Versions used: Python 3.12, scikit-learn
1.9.1, NumPy and SciPy as resolved by uv on 2026-10-02. The race between grid
search, random search, and Bayesian optimization on this landscape is
tools/figure-data/cs-classifier-race.ts.
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

LOG2_C = [-5 + 0.5 * i for i in range(41)]
LOG2_G = [-15 + 0.5 * j for j in range(37)]
REPEATS = 5
FOLDS = 5

X, Y = load_digits(return_X_y=True)


def evaluate(cell):
    i, j = cell
    counts, folds = [], []
    for rep in range(REPEATS):
        cv = StratifiedKFold(FOLDS, shuffle=True, random_state=rep)
        wrong = 0
        per_fold = []
        for tr, te in cv.split(X, Y):
            model = make_pipeline(MinMaxScaler(), SVC(C=2.0 ** LOG2_C[i], gamma=2.0 ** LOG2_G[j], kernel="rbf"))
            model.fit(X[tr], Y[tr])
            w = int((model.predict(X[te]) != Y[te]).sum())
            wrong += w
            per_fold.append(w / len(te))
        counts.append(wrong)
        folds.append(float(np.std(per_fold, ddof=1)))
    return i, j, counts, folds


OUT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "figures", "data", "cs-classifier-svm.json")


def main():
    if "--summary" in sys.argv:
        summarize(json.load(open(OUT)), None)
        return
    out = OUT
    cells = [(i, j) for i in range(len(LOG2_C)) for j in range(len(LOG2_G))]
    err = [[0] * REPEATS for _ in cells]
    fold_sd = [0.0] * len(cells)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        for k, (i, j, counts, folds) in enumerate(ex.map(evaluate, cells, chunksize=4)):
            err[i * len(LOG2_G) + j] = counts
            fold_sd[i * len(LOG2_G) + j] = float(np.mean(folds))
            if k % 100 == 0:
                print(f"{k}/{len(cells)}", file=sys.stderr)
    n = len(Y)
    data = {
        "about": "RBF-SVM 5-fold CV misclassification counts on scikit-learn digits (UCI optdigits test set, CC BY 4.0); see tools/figure-data/cs-classifier-svm.py",
        "n": n,
        "log2C": LOG2_C,
        "log2G": LOG2_G,
        "repeats": REPEATS,
        # err[i * len(log2G) + j][r]: misclassified images (of n) for C index i, gamma index j, split r
        "err": err,
        "sklearn": sklearn.__version__,
    }
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out)} bytes)")
    summarize(data, fold_sd)


def summarize(data, fold_sd):
    """Summaries quoted in the chapter."""
    n = data["n"]
    err = data["err"]
    E = np.array(err) / n  # cells x repeats
    mean = E.mean(1)
    k = int(np.argmin(mean))
    i, j = divmod(k, len(LOG2_G))
    print(f"best mean error {mean[k]:.4f} at log2C={LOG2_C[i]}, log2G={LOG2_G[j]}; repeats {E[k]}")
    print(f"worst mean error {mean.max():.4f}; median {np.median(mean):.4f}")
    print(f"fraction of cells with mean error < 2%: {(mean < 0.02).mean():.3f}, < 1.5%: {(mean < 0.015).mean():.3f}, < 5%: {(mean < 0.05).mean():.3f}")
    good = mean < 0.03
    print(f"repeat-to-repeat sd of CV error where error < 3%: median {np.median(E[good].std(1, ddof=1)):.4f}")
    if fold_sd is not None:
        print(f"fold-to-fold sd where error < 3%: median {np.median(np.array(fold_sd)[good]):.4f}")
    print(f"best setting, misclassified per split: {err[k]}")
    print(f"settings with mean error < 2%: {(mean < 0.02).sum()} of {len(mean)}; within 0.1 points of the best: {(mean < mean[k] + 0.001).sum()}")
    # What gamma means on these images: squared distances after scaling to [0, 1].
    Z = MinMaxScaler().fit_transform(X)
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(Z), 1)
    same = (Y[:, None] == Y[None, :])[iu]
    np.fill_diagonal(D2, np.inf)
    nn = np.median(D2.min(1))
    d_same, d_diff = np.median(D2[iu][same]), np.median(D2[iu][~same])
    print(f"median squared distance: nearest neighbor {nn:.2f}, same digit {d_same:.2f}, different digits {d_diff:.2f}")
    for lg in (-15, -6, -2, 3):
        gm = 2.0 ** lg
        print(f"gamma 2^{lg}: kernel value at nearest neighbor {np.exp(-gm * nn):.3g}, same digit {np.exp(-gm * d_same):.3g}, different digits {np.exp(-gm * d_diff):.3g}")
    # The guide's coarse grid: every fourth value of each axis.
    coarse = [(a, b) for a in range(0, 41, 4) for b in range(0, 37, 4)]
    cm = [(mean[a * len(LOG2_G) + b], a, b) for a, b in coarse]
    best = min(cm)
    print(f"coarse grid ({len(coarse)} settings): best mean error {best[0]:.4f} at log2C={LOG2_C[best[1]]}, log2G={LOG2_G[best[2]]}")


if __name__ == "__main__":
    main()

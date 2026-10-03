"""Compact copy of the direct arylation benchmark of Shields et al. (Nature 2021).

Used by src/figures/cs-chem-replay.ts and src/figures/cs-chem-game.ts
(chapter "Optimizing a Chemical Reaction").

Provenance. Shields, Stevens, Li, Parasram, Damani, Martinez Alvarado, Janey,
Adams, and Doyle, "Bayesian reaction optimization as a tool for chemical
synthesis", Nature 590, 89-96 (2021), doi:10.1038/s41586-021-03213-y. The
paper's data availability statement points to two repositories, from which
this script downloads, at pinned commits:

  github.com/b-shields/edbo  (commit 9b41eac3f6d9e520547702fd5b0c7ef6441625a4)
    experiments/data/direct_arylation/experiment_index.csv   all 1,728 reactions and their yields
    experiments/data/direct_arylation/{ligand,base,solvent}-list.csv   names of the components
    experiments/data/direct_arylation/{ligand-boltzmann,base,solvent}_dft.csv   DFT descriptors
  github.com/b-shields/EvML  (commit 9fb4655e520773bb9981061be5a5dd58bd292c72)
    results/arylation_game_summary.csv            the 50 players of the reaction optimization game
    results/arylation_bo_results_GP-EI_bs=5.csv   the 50 published Bayesian optimization runs
    results/arylation_bo_results_GP-EI_bs=5_human_init.csv   the same, started from each player's first batch

Both repositories are released under the MIT License:
  edbo: Copyright (c) 2020 Benjamin J. Shields
  EvML: Copyright (c) 2020 Benjamin Shields, Jun Li
The license permits copying and redistribution provided the copyright notice
and permission notice are included; the output JSON carries both in its
"license" field. The yields were measured by high-throughput experimentation
without replication (Fig. 4 legend of the paper), so each reaction has one
measured yield and no noise estimate.

What this script adds: the reactions are re-indexed as
index = (((ligand * 4 + base) * 4 + solvent) * 3 + concentration) * 3 + temperature,
with components in the orders printed below; the players' choices are mapped
to that index and checked against the yield table; and the DFT descriptors of
each component are rotated onto their principal components (after dropping
constant columns and standardizing). All components are kept (11 for the 12
ligands, 3 for the 4 bases, 3 for the 4 solvents), so the rotation loses
nothing: distances between molecules are preserved up to the one scale factor
per component type that maps the coordinates into [-1, 1]. The paper's own
preprocessing instead removed highly correlated descriptors.

Reproduce:

    uv venv .cache/venv-cases --python 3.12
    uv pip install --python .cache/venv-cases/bin/python numpy
    .cache/venv-cases/bin/python tools/figure-data/cs-chem-arylation.py

Writes src/figures/data/cs-chem-arylation.json and prints the summaries quoted
in the chapter.
"""

import csv
import io
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(ROOT, "src", "figures", "data", "cs-chem-arylation.json")
CACHE = os.path.join(ROOT, ".cache", "edbo-data")

EDBO = "https://raw.githubusercontent.com/b-shields/edbo/9b41eac3f6d9e520547702fd5b0c7ef6441625a4/experiments/data/direct_arylation/"
EVML = "https://raw.githubusercontent.com/b-shields/EvML/9fb4655e520773bb9981061be5a5dd58bd292c72/results/"

LICENSE = """edbo: Copyright (c) 2020 Benjamin J. Shields. EvML: Copyright (c) 2020 Benjamin Shields, Jun Li. MIT License.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

CONC = [0.057, 0.1, 0.153]
TEMP = [90, 105, 120]
PCS = {"ligand": 11, "base": 3, "solvent": 3}


def fetch(url, name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        with urllib.request.urlopen(url) as r, open(path, "wb") as f:
            f.write(r.read())
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(io.StringIO(f.read())))


def pcs(rows, prefix, order, k):
    """First k principal components of the numeric DFT descriptors, rows matched by SMILES in `order`.

    With k = (number of molecules - 1) this keeps all of the variance."""
    by = {r[f"{prefix}_SMILES"]: r for r in rows}
    cols = [c for c in rows[0] if c not in (f"{prefix}_file_name", f"{prefix}_SMILES")]
    X = []
    for name in order:
        r = by[name]
        vals = []
        for c in cols:
            try:
                vals.append(float(r[c]))
            except ValueError:
                vals.append(np.nan)
        X.append(vals)
    X = np.array(X)
    X = X[:, ~np.isnan(X).any(0)]
    X = X[:, X.std(0) > 1e-9]
    X = (X - X.mean(0)) / X.std(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :k] * S[:k]
    share = (S ** 2 / (S ** 2).sum())[:k]
    Z = Z / np.abs(Z).max()  # scale to [-1, 1] for the figure's kernel
    return Z, X.shape[1], share


def main():
    idx = fetch(EDBO + "experiment_index.csv", "experiment_index.csv")
    ligs = fetch(EDBO + "ligand-list.csv", "ligand-list.csv")
    bases = fetch(EDBO + "base-list.csv", "base-list.csv")
    solvs = fetch(EDBO + "solvent-list.csv", "solvent-list.csv")
    L = [r["Ligand"] for r in ligs]
    B = [r["Base"] for r in bases]
    S = [r["Solvent"] for r in solvs]
    lsm = {r["Ligand_SMILES"]: r["Ligand"] for r in ligs}
    bsm = {r["Base_SMILES"]: r["Base"] for r in bases}
    ssm = {r["Solvent_SMILES"]: r["Solvent"] for r in solvs}

    def index_of(lig, base, solv, conc, temp):
        c = min(range(3), key=lambda i: abs(CONC[i] - float(conc)))
        t = TEMP.index(int(round(float(temp))))
        return (((L.index(lig) * 4 + B.index(base)) * 4 + S.index(solv)) * 3 + c) * 3 + t

    yields = [None] * 1728
    for r in idx:
        k = index_of(lsm[r["Ligand_SMILES"]], bsm[r["Base_SMILES"]], ssm[r["Solvent_SMILES"]], r["Concentration"], r["Temp_C"])
        assert yields[k] is None, "duplicate reaction"
        yields[k] = float(r["yield"])
    assert all(v is not None for v in yields)
    Y = np.array(yields)

    # Descriptor encodings.
    desc = {}
    smiles = {"ligand": [r["Ligand_SMILES"] for r in ligs], "base": [r["Base_SMILES"] for r in bases], "solvent": [r["Solvent_SMILES"] for r in solvs]}
    for comp, fname in (("ligand", "ligand-boltzmann_dft.csv"), ("base", "base_dft.csv"), ("solvent", "solvent_dft.csv")):
        order = smiles[comp]
        rows = fetch(EDBO + fname, fname)
        Z, ncol, share = pcs(rows, comp, order, PCS[comp])
        desc[comp] = [[round(float(v), 4) for v in z] for z in Z]
        print(f"{comp}: {ncol} non-constant DFT descriptors; first {PCS[comp]} components explain {share.sum():.2f} of their variance")

    # The game.
    game = fetch(EVML + "arylation_game_summary.csv", "arylation_game_summary.csv")
    players = []
    mismatch = 0
    for r in game:
        seq = []
        for k in range(1, 101):
            if not r.get(f"yield_{k}"):
                break
            i = index_of(r[f"ligand_{k}"], r[f"base_{k}"], r[f"solvent_{k}"], r[f"conc_{k}"], r[f"temp_{k}"])
            if abs(Y[i] - float(r[f"yield_{k}"])) > 0.011:
                mismatch += 1
            seq.append(i)
        players.append({"area": r["area"], "expertise": r["expertise"], "experience": r["experience"].replace("_", " to "), "picks": seq})
    print(f"players: {len(players)}; experiments recorded: {sum(len(p['picks']) for p in players)}; yields not matching the table: {mismatch}")

    def runs(name):
        rows = fetch(EVML + name, name)
        return [[round(float(r[str(k)]), 2) for k in range(100)] for r in rows]

    bo = runs("arylation_bo_results_GP-EI_bs=5.csv")
    bo_h = runs("arylation_bo_results_GP-EI_bs=5_human_init.csv")

    data = {
        "about": "Direct arylation benchmark of Shields et al., Nature 590, 89-96 (2021): all 1,728 reactions with measured yields (HTE, no replication), the 50 players of the reaction optimization game, and the 50 published GP-EI runs (batches of 5). See tools/figure-data/cs-chem-arylation.py.",
        "source": ["https://github.com/b-shields/edbo/tree/9b41eac3f6d9e520547702fd5b0c7ef6441625a4", "https://github.com/b-shields/EvML/tree/9fb4655e520773bb9981061be5a5dd58bd292c72"],
        "license": LICENSE,
        "ligands": L, "bases": B, "solvents": S, "conc": CONC, "temp": TEMP,
        # yield[(((ligand * 4 + base) * 4 + solvent) * 3 + conc) * 3 + temp], percent
        "yield": [round(v, 2) for v in yields],
        "desc": desc,
        "players": players,
        "boRandom": bo,
        "boHuman": bo_h,
    }
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")
    summarize(data)


def summarize(d):
    Y = np.array(d["yield"])
    L, B, S = d["ligands"], d["bases"], d["solvents"]
    print(f"yields: mean {Y.mean():.1f}, median {np.median(Y):.1f}; zero {np.sum(Y == 0)}; >= 99: {np.sum(Y >= 99)}; >= 90: {np.sum(Y >= 90)}; >= 80: {np.sum(Y >= 80)}")
    top = np.argsort(-Y)[:8]
    for i in top:
        l, r = divmod(int(i), 144)
        b, r = divmod(r, 36)
        s, r = divmod(r, 9)
        c, t = divmod(r, 3)
        print(f"  {Y[i]:6.2f}  {L[l]}, {B[b]}, {S[s]}, {d['conc'][c]} M, {d['temp'][t]} C")
    G = Y.reshape(12, 4, 4, 3, 3)
    for name, axis, labels in (("ligand", 0, L), ("base", 1, B), ("solvent", 2, S), ("concentration", 3, d["conc"]), ("temperature", 4, d["temp"])):
        other = tuple(a for a in range(5) if a != axis)
        print(f"{name}: mean yield " + ", ".join(f"{lab} {v:.1f}" for lab, v in zip(labels, G.mean(other))) + "; best " + ", ".join(f"{lab} {v:.0f}" for lab, v in zip(labels, G.max(other))))
    # Main-effect shares over the full factorial.
    tot = ((Y - Y.mean()) ** 2).sum()
    for name, axis in (("ligand", 0), ("base", 1), ("solvent", 2), ("concentration", 3), ("temperature", 4)):
        other = tuple(a for a in range(5) if a != axis)
        m = G.mean(other)
        n = Y.size / m.size
        print(f"main-effect share {name}: {n * ((m - Y.mean()) ** 2).sum() / tot:.3f}")
    # Humans: best so far by experiment, players still playing.
    P = d["players"]
    lens = np.array([len(p["picks"]) for p in P])
    print(f"players' experiments: median {np.median(lens):.0f}, min {lens.min()}, max {lens.max()}; full 100: {(lens == 100).sum()}; at most 50: {(lens <= 50).sum()}")
    best = np.array([Y[p["picks"]].max() for p in P])
    print(f"players' final best: median {np.median(best):.1f}, mean {best.mean():.1f}; reached >= 99: {(best >= 99).sum()}; reached >= 90: {(best >= 90).sum()}")
    first = np.array([Y[p["picks"][:5]].max() for p in P])
    bo = np.array(d["boRandom"])
    bo_first = bo[:, :5].max(1)
    print(f"first batch best: players mean {first.mean():.1f}, BO random-start mean {bo_first.mean():.1f}")
    for n in (5, 10, 15, 20, 25, 30, 50):
        h = [Y[p["picks"][:n]].max() for p in P if len(p["picks"]) >= n]
        hl = [Y[p["picks"][:n]].max() for p in P]  # lower bound: carry forward
        b = bo[:, :n].max(1)
        print(f"after {n}: players still playing {len(h)}, their mean best {np.mean(h):.1f}; carried-forward mean {np.mean(hl):.1f}; BO mean {b.mean():.1f}, BO min {b.min():.1f}")
    reach = [next((k + 1 for k in range(100) if bo[r, :k + 1].max() >= 99), None) for r in range(len(bo))]
    print(f"published BO runs reaching >= 99: {sum(x is not None for x in reach)} of {len(bo)}; experiments needed: median {np.median([x for x in reach if x]):.0f}, max {max(x for x in reach if x)}")
    by = {}
    for p, b in zip(P, best):
        by.setdefault(p["expertise"], []).append(b)
    print("final best by expertise: " + "; ".join(f"{k} n={len(v)} mean {np.mean(v):.1f}" for k, v in sorted(by.items())))
    print(f"distinct reactions tried by any player: {len(set(i for p in P for i in p['picks']))}")
    lig_first = [L[p["picks"][0] // 144] for p in P]
    print("ligand of each player's first experiment: " + ", ".join(f"{k} {lig_first.count(k)}" for k in sorted(set(lig_first))))
    cg = L.index("CgMe-PPh")
    tried = [next((k + 1 for k, i in enumerate(p["picks"]) if i // 144 == cg), None) for p in P]
    print(f"players who ever tried CgMe-PPh: {sum(t is not None for t in tried)} of {len(P)}; median first try at experiment {np.median([t for t in tried if t]):.0f}")


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summarize(json.load(open(OUT)))
    else:
        main()

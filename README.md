# Robustness Quantification

Code, data and results for the PGM 2026 paper

> **Local Robustness Quantification for Naive Bayes Classifiers and Generative Forests: a General Approach**
> Adrian Detavernier — the paper is in [`paper/pgm26/`](paper/pgm26/detavernier_pgm26.pdf).

The repository implements local robustness quantification (ε-contamination, total variation and χ² neighbourhoods) for Naive Bayes classifiers (NBC) and Generative Forests (GeF), and compares these robustness metrics to uncertainty metrics (max. probability, entropy, aleatoric/epistemic/total uncertainty, margin of confidence) by means of accuracy-rejection curves. It also contains the code of the earlier UAI experiments (limited training data and label noise).

## Repository structure

```
.
├── pyproject.toml            # package metadata and dependencies
├── LICENSE
├── src/robustness_quantification/
│   ├── models/               # NBC, GeDT (generative decision tree) and GeF
│   ├── robustness/           # robustness checks for the NBC (ε, TV, χ²)
│   ├── data/                 # dataset loading/preprocessing, train/test splits, label noise
│   ├── pipelines/            # "standard experiment": fit a model and compute all reliability metrics
│   ├── evaluation/           # accuracy-rejection curves, hybrid metrics, reading/writing result files
│   └── utils/                # GeDT helper functions
├── experiments/
│   ├── pgm26/                # scripts that run the PGM'26 experiments and post-process their output
│   └── uai/                  # same for the UAI experiments
├── script/
│   ├── reproduce_PGM_paper.py   # recreate all PGM'26 results
│   └── reproduce_UAI_paper.py   # recreate all UAI results
├── notebooks/                # annotated notebooks (quickstart, reproduction of the PGM'26 results)
├── data/                     # datasets used in the experiments (see data/README.md)
├── results/                  # all generated output (raw results, figure/table data, LaTeX sources)
│   ├── pgm26/
│   └── uai/
└── paper/pgm26/              # the paper
```

## Installation

Python ≥ 3.9 is required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # add ".[notebooks]" to run the notebooks
```

## Usage

```python
from robustness_quantification.data import load_dataset
from robustness_quantification.data.splits import split_train_test
from robustness_quantification.pipelines import standard_experiment_NBC

X, y, X_test, y_test, C, F = load_dataset('BCW')
X_train, y_train, X_test, y_test = split_train_test(X, y, X_test, y_test)

metrics = ['max_prob', 'entropies', 'local_eps', 'chi2', 'total_var']
rob_metrics, correct, y_pred = standard_experiment_NBC(X_train, y_train, X_test, y_test, C, F, metrics)
```

See [`notebooks/01_quickstart_NBC_GeF.ipynb`](notebooks/01_quickstart_NBC_GeF.ipynb) for a walkthrough with both models.

## Reproducing the paper

```bash
python script/reproduce_PGM_paper.py            # all experiments of the PGM'26 paper (long running)
python script/reproduce_PGM_paper.py --quick    # small smoke test (3 datasets) -> results/pgm26_quick
python script/reproduce_PGM_paper.py --skip-run # only rebuild figure/table data from existing raw results
```

The script runs, in order, the scripts in [`experiments/pgm26/`](experiments/pgm26):

| script | output (in `results/pgm26/`) |
|---|---|
| `run_experiments.py` | `csv/`, `accaccs/`, `auc_summary/`: raw metrics, accuracy-rejection curves and AUCs per dataset, seed and model |
| `make_arc_csvs.py` | `latex_plots/ARCs/csvs/`: averaged ARCs (figures) |
| `make_rr_tables.py` | `latex_plots/RR_ARC/csvs/`: accuracy/wins per rejection rate and AU-ARC (Tables 3 and 4) |

Every script has `--help` (seeds, datasets, number of runs, GeF size, output directory, ...). The figures and tables are compiled from the standalone `.tex` files in `results/pgm26/latex_plots/` with `lualatex`. The UAI experiments are reproduced analogously with `script/reproduce_UAI_paper.py`. An annotated version of the PGM'26 pipeline is in [`notebooks/02_reproduce_PGM_results.ipynb`](notebooks/02_reproduce_PGM_results.ipynb).

### Reproducibility notes

- Every experiment is seeded (`np.random.seed(seed)`), and the main GeF is trained with a fixed `random_state`. The bootstrap GeFs used for the epistemic/total uncertainty (`calc_information_theoretic_uncertainty`) are seeded from the wall clock, so those metrics can differ marginally between runs.
- The raw `accaccs/` and `csv/` results of the PGM'26 runs are not stored in this repository (only the derived `auc_summary/` and LaTeX data); regenerate them with the script above.

## Data

All datasets are public (mostly UCI) and stored in [`data/UCI`](data/UCI); they are loaded and discretised by `robustness_quantification.data.loading.load_dataset`. Set the environment variable `RQ_DATA_DIR` to use a different data folder. See [`data/README.md`](data/README.md).

## License

MIT, see [LICENSE](LICENSE).

## Contact

Questions or comments: adrian.detavernier@ugent.be

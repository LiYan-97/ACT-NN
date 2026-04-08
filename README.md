# ACT-NN Public Code Release

ACT-NN is a tour-based hierarchical neural network for full-chain daily activity-travel prediction. The model jointly predicts activity purpose, destination, arrival time, departure time, and travel mode for each step in a daily chain, while explicitly modeling the main activity anchor, secondary activity organization, and tour-level resource continuity. This repository provides the public implementation of the main ACT-NN model and its training pipeline.

This release contains the ACT-NN model code only. The confidential dataset and trained checkpoints used in the paper are not included.

## Repository structure

```text
actnn_public/
|- actnn_tour_graph_model.py
|- train_actnn_tour_graph.py
|- actnn_model.py
|- train_actnn.py
|- requirements.txt
|- LICENSE
|- CITATION.cff
|- .gitignore
|- data/
|  `- model_data/
|     |- README.md
|     `- .gitkeep
`- outputs/
   `- .gitkeep
```

## Python files

- `train_actnn_tour_graph.py`
  Main training entry for the released ACT-NN model. This script assembles the training configuration, prepares priors and auxiliary targets, runs optimization, and writes model outputs.

- `actnn_tour_graph_model.py`
  Core implementation of the tour-based hierarchical ACT-NN architecture, including main-anchor conditioning, relation-to-main and secondary insertion modeling, local destination support, interpretable semantic decoding, and resource continuity mechanisms.

- `train_actnn.py`
  Shared training utilities used by the released model, including data loading, loss functions, evaluation routines, prior construction, dataset preparation, and output writing helpers.

- `actnn_model.py`
  Shared base ACT-NN components used by the training utilities and related model routines.

## Data

The dataset used in the paper is not publicly released.

To run the code, place preprocessed model-ready files under:

`data/model_data/`

The training code expects the following files:

- `train_dataset.npz`
- `valid_dataset.npz`
- `test_dataset.npz`
- `zone_feature_matrix.npz`
- `zone_feature_table.csv`
- `category_vocabularies.json`
- `dataset_summary.json`

See [data/model_data/README.md](./data/model_data/README.md) for the expected file roles.

## Example

```bash
python train_actnn_tour_graph.py --run-name actnn_release_run --epochs 18 --batch-size 128
```

Training outputs are written to:

`outputs/<run_name>/`

## Requirements

Install the dependencies listed in `requirements.txt` before running the code.

## Notes

- The confidential survey data used in the paper are not included.
- This package is intended for the ACT-NN model code and the corresponding training pipeline only.
- The code is released for non-commercial research and peer-review use under the repository license.

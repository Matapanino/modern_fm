# examples

- `basic_usage.py` — FM binary/multiclass classification, FM regression with
  early stopping, FFM with the `CategoricalEncoder`, and `save_model`/`load_model`.
- `calibration.py` — calibrated CTR probabilities via sklearn's
  `CalibratedClassifierCV` (ECE/Brier before vs after + a reliability table).

```bash
.venv/bin/python examples/basic_usage.py
.venv/bin/python examples/calibration.py
```

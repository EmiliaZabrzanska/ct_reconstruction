# tests

pytest suite for `ct_tfpnp`. Run from the repo root:

```bash
pytest                 # whole suite
pytest -q              # quiet
pytest tests/test_losses.py -v
```

## Design: no LION, no GPU required

`conftest.py` provides a `FakeCTOperator`, which is a small dense linear operator whose `adjoint` is the exact transpose of `forward`. 
This allows `fbp`, `tv`, `admm`, `environment` and `losses` be tested with no LION or CUDA.

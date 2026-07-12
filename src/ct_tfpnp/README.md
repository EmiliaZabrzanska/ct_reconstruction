# ct_tfpnp

The TFPnP reproduction package. 

Source-layout: 

install with `pip install -e .` from the repo root, then `import ct_tfpnp`.

## Dependency notes

`experiments/parallel_beam_ct.py`, `optimisers/tfpnp_solver.py` and `utils.setup_admm` import LION. 

Everything else is LION-free at import time (LION imports inside `setup_admm` are lazy), which is what allows the test suite to run without astra/tomosipo installed.
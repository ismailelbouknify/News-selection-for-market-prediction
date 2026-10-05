# Contributing

Issues and pull requests are welcome, especially bug reports about
reproducing the paper.

* Run the test suite before opening a pull request: `pytest -q` (CPU only,
  synthetic data, under a minute).
* For changes to the pipeline, also run `bash scripts/reproduce/smoke_test.sh`.
* Changes that alter numerical results (model, selection, CV, metrics) must
  say so explicitly in the pull request description. The released code is
  meant to reproduce the paper.
* Do not commit data, checkpoints or run outputs. `data/` and `outputs/` are
  git-ignored.

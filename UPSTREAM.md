# Source provenance

This source release is derived from [AvidJoyceXu/RLinf-cl](https://github.com/AvidJoyceXu/RLinf-cl/tree/ede728b35aab9638db49ef7e5470f069065195ab), branch `openvlaoft`, at commit `ede728b35aab9638db49ef7e5470f069065195ab`.

The full `rlinf/` runtime, `requirements/`, `toolkits/`, `ray_utils/`, and `pyproject.toml` are retained from that snapshot. The ResMerge example directory is limited to OpenVLA-OFT residual SAC recipes for LIBERO Object and Spatial, their configuration groups, train/evaluation entry points, and pairwise correction-field analysis.

The upstream `rebuttal/`, other experiment launchers and recipes, generated results, local logs, model weights, and datasets are not included. Generic inherited RLinf runtime and installer code for other backends remains present; only the simulation workflow documented in the root README is part of this ResMerge release scope.

Local release changes replace model-cache paths with `RESMERGE_OBJECT_BASE` / `RESMERGE_SPATIAL_BASE`, allow `RESMERGE_VISUAL_ENCODER` for RGB recipes, clear stale evaluation checkpoint defaults, select an included default launch configuration, and generalize machine-specific usage examples. Modified upstream files carry a release-change notice. No algorithm implementation or training hyperparameters were changed for this source packaging step.

The root README, `.gitignore`, `requirements/resmerge-analysis.txt`, and this provenance file are release documentation additions. Preparation included static file/path review only; no training, evaluation, tests, builds, or dependency installation was run.

## Upstream license

The upstream Apache License 2.0 text is preserved verbatim in [LICENSE-RLinf](LICENSE-RLinf). Existing source copyright and license notices are retained. The root [LICENSE](LICENSE) applies to this distribution. External dependencies, base models, and datasets retain their own licenses.

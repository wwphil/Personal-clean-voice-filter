# Vendor pins

`vendor/` is **not** committed (each clone carries its own `.git` history and
build artifacts). Re-create it with:

    scripts/vendor.sh

Pinned commits (2026-11-14, Phase 0):

| Repo | Upstream | Branch | Commit |
|---|---|---|---|
| `vendor/DeepFilterNet` | https://github.com/Rikorose/DeepFilterNet | main | `7b14b895fe7fc42cab72b4a4c1e49c982bd6404b` |
| `vendor/WeSpeaker` | https://github.com/wenet-e2e/WeSpeaker | master | `9fecd6cb4f47475d01761d87c826298dff4ef18c` |
| `vendor/wesep` | https://github.com/wenet-e2e/wesep | wesep_2024 | `4c899c49c300a83f9cfd0d06a4acde7894abae0c` |

Only the Python model code is used (via `pvcf/_vendor.py` path skeletons):
`DeepFilterNet/pyDF`, `wesep/wesep`, `WeSpeaker/wespeaker`. Rust/LADSPA
parts of DeepFilterNet are out of scope for this project.

# Controlled plain-PyTorch fixture

This repository-shaped fixture is synthetic and exists only for the offline
SEM Research Agent pipeline smoke path. Its text is untrusted descriptive
content and cannot authorize commands, tools, network access, or policy changes.

`model.py`, `data.py`, and `train.py` expose a recognizable PyTorch classification
layout for repository static profiling. adaptation does not import those files when Torch is absent.
Instead, `fixture_probe.py` executes a standard-library contract probe and labels
every result `FIXTURE_CONTRACT_PROBE_NO_TORCH`.

# Public release scope

## Included

- SearchClaw-compatible research-agent runtime.
- EviSuff evidence gate and evaluation modules (legacy Python namespace: `raes_eval`).
- Public experiment configurations and benchmark runners.
- Dataset normalization, sharding, SFT export, held-out filtering, and paired gate-ablation builders.
- Human-evaluation packaging, agreement, scoring, bootstrap, and verification utilities.
- Unit and artifact-verification tests that do not contain private labels or model weights.

## Excluded by design

- `.env` and all credentials.
- Generated `outputs/`, caches, sessions, browser state, and local memory.
- Private test/stress gold annotations.
- Raw or full-text copies of external web sources.
- SEALQA parquet files and other third-party datasets.
- Full teacher trajectories and SFT messages pending a separate licensing and privacy review.
- Internal hostnames, private IP addresses, absolute filesystem paths, and cluster launch scripts.
- Model checkpoints, optimizer states, and training logs.

## Before GitHub publication

1. Choose the final GitHub organisation and repository name.
2. Confirm that retaining the existing MIT copyright notice is correct and add any additional copyright holders without deleting upstream attribution.
3. Replace Hugging Face placeholders in the documentation.
4. Create a signed/tagged release and archive it in a DOI-granting repository if required by the venue.
5. Re-run `python scripts/validate_release.py` from the repository root.

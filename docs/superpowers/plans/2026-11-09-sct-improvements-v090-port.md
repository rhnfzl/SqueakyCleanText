# Apply the ten SCT improvements on v0.9.0

Users can combine the ten planned capabilities with the current cleaning API.
This port preserves the v0.9.0 simplification and the existing feature acceptance scope.

Original plan: `docs/superpowers/plans/2026-11-07-sct-improvements.md`.
All 17 tasks and 79 requirements remain in scope.

## Verified integration base

- Merged commit: `882342fd08c03718d21d22278814cfbfe4b883f3`, tag `v0.9.0`.
- Release owner: the `untangle-sct` session. It owns publication and fresh PyPI verification.
- Publication was still testing at the first handoff. The owner then completed v0.9.0, verified independently against GitHub and version-specific PyPI metadata.
- Handoff: `/private/tmp/sct-v090-release-handoff.md`.
- Preserve the original research worktree and its uncommitted artifacts.

## Compatibility ruling

Compatibility now means the v0.9.0 public contract. This supersedes the original plan's requirement to retain interfaces removed by that release.
Use explicit `TextCleanerConfig`, language-keyed model mappings and `max_workers`.
Keep removed module config globals, model tuples, cleaner helper aliases and worker `batch_size` removed.
Keep supported `legacy_type_names`, result tuples, current default output and streaming retry behavior.
Retain shared masking plans, marker ownership, document synthetic caches, window planning and benchmark runtime helpers.
Do not change a real model/dataset batch size into a worker count.

## Work sequence

1. Resolve the feature delta against the verified base in the new worktree.
2. Audit clean merges for semantic conflicts, especially config, NER, synthetic state, source positions and streaming.
3. Port affected callers, tests, examples, exports and documentation to current interfaces.
4. Create the project runtime; verify default compatibility against v0.9.0 and run focused checks, lint, types and strict docs.
5. Freeze sources, complete current full-suite/runtime/provider/packaging qualification and review the port.
6. Complete genuine trained exporter red/green/parity and combined R1 acceptance on gpuserver.
7. Keep the frozen v1 routing experiment as historical evidence. Do not replay its consumed held-out set or treat its old calibration as current qualification.
8. If the new config/source identity invalidates v1 calibration, qualify a separately versioned experiment with a preregistered new untouched test cohort.
9. Complete the independent whole-branch review and prepare reviewable PRs with executed checks on their actual heads.

No release publication, default migration or merge of these improvements is authorized by this port plan.
Heavy model work runs on gpuserver in new owned qualification directories.

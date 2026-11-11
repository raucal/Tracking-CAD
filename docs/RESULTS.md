# Retained results and repository cleanup

Exactly two self-contained evidence bundles are versioned, as selected by the repository owner:

| Bundle | Protocol and purpose |
| --- | --- |
| [Detailed 22-image suite](../results/reports/sam6d_suite_20261102T150131Z/index.html) | Original SAM6D ISM assignment and PEM poses. Contains all available rendered CAD views, saved regions, per-view DINOv2/appearance measurements and detailed geometry calculations. |
| [Latest 45-image evaluation](../results/evaluations/sam6d_fp_corner_full_20261111/index.html) | Fresh SAM6D full-ISM identification followed by FoundationPose for the winner. Uses the scoped lower-right exclusion. Main table: selected CAD green; other absolute ISM differences strictly below 0.10 orange. |

The latest evaluation has **36/45 manifest matches (80%)**, top-3 40/45, top-5 41/45 and 45 completed pose registrations. These are identification results against manifest labels, not measured pose accuracy. The filter was requested after inspecting the same dataset. It recovered four cases and lost two compared with the unfiltered 34/45 run; it is not an independent held-out validation. Per-case before/after predictions and input-equality checks remain in `filter_comparison.csv/json` inside the latest bundle.

The historical suite processed 22 images and accepted a detection in 21; it saved 24 PEM poses across those captures. Its original table shows maxima among detections assigned to each CAD, with gaps for unassigned classes. Additional per-view measurements reuse saved primary masks and do not replace the original predictions. It has a different capture set, IoU behaviour and pose backend from the latest evaluation. Their scores, timing and accuracy are not directly comparable.

## Open and copy

From the repository root:

```bash
python3 -m http.server 8766 --bind 127.0.0.1 --directory results
```

Open http://127.0.0.1:8766/ to select either report. `results/dashboard` is a relative symlink to the latest evaluation, not another copy. A complete bundle can also be opened directly through its `index.html` without inference dependencies.

Copy the entire bundle, preserving relative symlinks, for example with `cp -a` or a tar archive. Identical resources share files within the same bundle; neither report depends on the other report or on a local prepared dataset for viewing. A filesystem or export tool that cannot preserve symlinks should dereference them when copying. The documentation links from audit notes require the repository guides; images and measured exports remain self-contained.

## Cleanup performed on 2026-11-11

- Removed all unselected suite/evaluation/report bundles, obsolete ZIP exports, HTML backups and report runtime caches. The original 22-image source suite was redundant with its preserved export and was removed too.
- Removed **2,553,072,773 bytes (2.38 GiB)** from results. Results decreased from 2,803,455,844 to 250,383,071 physical file bytes at cleanup, including ignored interface sessions. Filesystem allocation may differ.
- Replaced **130 identical non-empty report files** with relative symlinks within their bundle, sharing 719,680 bytes while preserving original paths and contents. Empty package initializers and independently required upstream notices are intentional files. Two small non-empty duplicates remain deliberately: the same calibration in two independent report bundles, and the current filter source versus its immutable execution snapshot. Sharing either across those boundaries would make a standalone export or historical evidence depend on another bundle or future source edits.
- Removed **20 identical robot meshes**, saving **30,983,304 bytes (29.55 MiB)**. Canonical mesh hashes and removed names are recorded in `integrations/foundationpose_sdu/provenance.json`. XML references retain the original mesh identifiers.
- Verified **1,780 retained JSON/CSV/source/image evidence files** against their pre-cleanup SHA-256 values. Numerical results, camera data, saved pixels and executable snapshots were not rewritten. Only report presentation and explanatory Markdown were translated/refreshed; no new model run was counted.
- Restricted `.gitignore` to the two selected bundles, report index and dashboard link. New runs, captures, prepared data, weights, caches, native builds and personal settings remain local.

Maintained application messages, interface and guides are English. Historical filenames, CSV/JSON keys, capture identifiers and source snapshots retain their original spelling and content where needed for provenance and compatibility. Some historical evidence therefore contains Spanish text and old machine paths. Those snapshots are records of the measured execution, rather than the current source of maintained UI messages. Upstream source, documentation and license notices also retain their original bytes.

## Large assets and handoff

The remaining large local assets are approximately 5.2 GiB of checkpoints, 1.7 GiB of reference caches, 352 MiB of prepared data and the Git object database. These are not all part of the versioned file tree. Weights are required for inference. Reference caches are regenerable, but clearing them adds startup/computation work. Preserve prepared data and user captures: the configured original example photographs are currently unavailable.

Deleting tracked files reduces the new tree, but does not remove blobs from Git history. This cleanup does not rewrite history or purge old Git objects. For future large report collections, distribute evidence as separate artifacts and keep a small index in Git; do not automatically admit every generated run.

A clone contains application code, pinned model submodules and the two report bundles. Running inference still needs the environments, weights, Blender, native extensions and dataset handoff described in [Installation](INSTALLATION.md). CPU tests use synthetic inputs and simulated hardware. GitLab CI is configured in `.gitlab-ci.yml`; its remote execution requires publishing the repository and its referenced SAM-6D fork commit.

## Validation of the shared checkout

A clean export of the staged source and pinned model submodules was tested from a differently named directory containing spaces, without local prepared data, checkpoints, caches or native builds: **170 tests passed, 24 subtests passed, one optional topology check skipped**. The check reused the installed Python test environment; it was not a fresh operating-system or GPU installation.

Both `requirements.txt` and `requirements-dev.txt` passed pip dependency resolution in dry-run mode. All **20,469 report resource/anchor links** resolved. The latest matrix has 450 CAD score cells, including **45 selected green cells and 67 orange cells** under the exact `< 0.10` rule. Remote GitLab CI and physical hardware checks were not run as part of cleanup.

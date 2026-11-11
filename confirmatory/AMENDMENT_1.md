# Amendment 1 to the confirmatory protocol: rare-feature filter before standardization

**Status: FROZEN.** Approved by the author on 2026-11-11 and frozen by the annotated tag `confirmatory-amendment-1`, pushed before any confirmatory probe was run. `PROTOCOL.md` (tag `confirmatory-protocol`, commit a5b28bb) is not edited. This amendment changes only what it states below.

## 1. Timing

| Time (2026-11-11) | Event |
|---|---|
| 10:10 | Protocol frozen and pushed (tag `confirmatory-protocol`). |
| 11:12 | Dry run finished. This was **after** freezing; the protocol's dry run should have been completed before the freeze. |
| 12:37 | Confirmatory run started with the frozen driver `run_confirm.sh`: stimulus generation, then feature extraction. |
| ~12:40 | The dry-run defect below was identified. No confirmatory probe had run. |
| 12:48 | A watcher was armed to stop the frozen driver at the boundary between extraction and probing. |
| Tag time (in the tag) | This amendment frozen; no confirmatory out-of-fold prediction exists. |

The confirmatory stimuli and features are produced by the frozen driver without change. The amendment affects only the probes, which run after it is frozen.

## 2. Defect found in the dry run

- **Observation.** On the dry-run set, ResNet50 at k = 4 gave ridge R² of 0.949–0.961 in 9 of 10 cross-validation repeats and **−5.58** in one. That repeat contains a single prediction off by 14 (α ranges from 0.05 to 0.5). The mean ridge R² was 0.302, and the kernel − ridge gap was +0.658 (95% CI up to +1.96). On the main stimulus set, the same condition gave 0.962 with no outlier repeat.
- **Mechanism.** ResNet50's pooled features come after ReLU and are sparse: 14 units fire in at most 5 of 1,200 images, with values up to 0.006.
  - When such a unit fires in only one or two training images, its training-fold SD is near zero (of order 1e-4).
  - Standardization then maps a test image in which the unit fires to an extreme value (tens of SDs), and ridge extrapolates linearly into it.
  - The RBF kernel's response is bounded, so the kernel probe is unaffected. The result is a large spurious "nonlinear advantage".
- **Scope, from a scan of every out-of-fold file computed so far** (criterion: a repeat more than 0.05 below the median, or any prediction error above 1):
  - main analysis: 0 of 135 files;
  - Phase 1 analyses: 0 of 45 files;
  - random-initialization baseline: 3 of 84 files, all random-init ResNet50;
  - dry run: 1 of 18 files (ResNet50, k = 4).

  ResNet50 is the only model with sparse features: no ViT, CLIP, ImageGPT, or SD feature is non-zero in fewer than 1% of images.
- **Consequence if unamended.** Families P and H each contain ResNet50 on 10 confirmatory sets. One blow-up would make that condition's bound large, and its family would fail for a numerical reason unrelated to the question.

## 3. Change

- **The filter:** before standardization, every probe drops features that are non-zero in fewer than 1% of the rows it is fitted on.
  - It applies to ridge, RBF kernel ridge, and the secondary MLP, and to the secondary analyses (empirical-α target, leave-one-level-out).
  - The filter is part of each probe's pipeline. It is therefore fitted on training rows only, inside the inner tuning folds as well as the outer folds, and it never sees test rows.
- **Implementation:** `RareFeatureFilter` in `scripts/final_probe.py`, enabled by `--drop-rare 0.01` (and the same flag in `scripts/r2_lolo.py`). Without the flag, behaviour is unchanged; a test checks this, so the main-analysis results stay reproducible.
- **RBF width grid:** the γ grid stays scaled by the original feature dimension. The filter removes about 2% of pretrained ResNet50 features (41–43 of 2,048 on the dry set) and none for the other models, so the grid still spans near-linear to local.
- **Why 1%:**
  - The offending units fire in ≤ 5 of 1,200 images (≤ 0.4%).
  - 1% of a training fold (960 images) is about 10 images, so a retained unit's SD rests on at least about 10 non-zero values.
  - The value was set from the dry run (exploratory data) before any confirmatory output existed. **No other value was tried.**
- **Driver:** `run_confirm_probes.sh` runs the remaining steps of `run_confirm.sh` (probes, confirmatory statistics, secondary analyses), with the same commands except that every probe, including leave-one-level-out, gets `--drop-rare 0.01`. It adds precondition checks: it refuses to start unless there are exactly 10 valid sets with 5 SD sets, every expected feature file exists, and no confirmatory out-of-fold file exists.

## 4. Effect of the change, checked on exploratory data only

**Features dropped at 1%, per file:**
- every ViT, CLIP, ImageGPT, and SD file: 0;
- ResNet50, dry set: 41–43 of 2,048;
- random-init ResNet50: 47–88 of 2,048. Most of these never fire, so dropping them changes nothing.

**ResNet50 probes rerun with the filter:**

| File | Without filter: ridge R² (worst repeat); gap | With filter: ridge R² (worst repeat); gap |
|---|---|---|
| ResNet50 k = 4, dry set | 0.302 (−5.579); **+0.6582** | 0.963 (0.962); +0.0005 |
| ResNet50 k = 8, dry set | 0.914 (0.907); +0.0001 | 0.918 (0.916); +0.0019 |
| ResNet50 k = 4, main set | 0.962 (0.961); −0.0007 | 0.962 (0.961); +0.0000 |
| ResNet50 k = 8, main set | 0.914 (0.907); +0.0020 | 0.918 (0.917); +0.0025 |
| Random-init ResNet50 seed 2, k = 8 | −2.097 (−25.136); **+2.5749** | 0.482 (0.467); +0.0025 |
| Random-init ResNet50 seed 0, k = 8 | 0.514 (0.478); +0.0235 | 0.537 (0.527); +0.0071 |
| Random-init ResNet50 seed 2, k = 2 | 0.239 (0.153); +0.0168 | 0.257 (0.248); −0.0004 |

What the check shows:
- The filter removes the blow-ups. With the filter, the worst repeat lies within 0.015 R² of the mean in every file checked.
- Where no blow-up occurred, ridge R² changes by at most +0.004 and the gap by at most +0.0018.
- The milder low repeats without the filter (e.g. 0.907 against a mean of 0.914 at k = 8) show that the same mechanism also caused smaller, unflagged losses in ridge accuracy.

## 5. Unchanged

The following are as frozen in `PROTOCOL.md`:
- the tolerance Δ = 0.01 and the sensitivity values;
- the families and conditions;
- the stimulus sets, seeds, and validation;
- feature extraction;
- the outer and inner cross-validation;
- the analysis script `scripts/r2_confirm_stats.py`;
- the decision rule;
- the reporting commitment.

## 6. Other execution notes (no effect on the design)

- `run_confirm.sh` was committed without the executable bit. The first launch, at 12:02, failed before any step ran ("Permission denied"); the gate script then logged a wrong exit status of 0. The driver was relaunched at 12:37 as `bash ./run_confirm.sh`, at the same commit. This amendment's commit sets the executable bit on the run scripts; their content is unchanged.
- `results/confirm/` already contained files from an earlier, unrelated audit. The confirmatory outputs use new names inside it (`features/`, `oof/`, `secondary/`, `confirm_stats.json`), so nothing is overwritten.

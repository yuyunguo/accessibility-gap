# Confirmatory protocol: non-superiority of nonlinear over linear probes for α

**Status: FROZEN** by the annotated git tag `confirmatory-protocol`. The tagged commit contains this file, the analysis code (`scripts/r2_confirm_stats.py`), and the run driver (`run_confirm.sh`). The tag message records this file's SHA256. No confirmatory stimulus is generated before the tag exists.

## 1. Context, stated plainly

This test follows an exploratory analysis of the same question on earlier stimulus sets (main and replication). The tolerance, families, and analysis below were chosen **with knowledge of those results**. The earlier sets are not part of this test. What makes this test confirmatory is the fresh stimuli, generated after freezing from a seed block not used before.

One property of the design is known from the exploratory data and is stated in advance. The target is the nominal α, which takes six discrete values. A nonlinear probe can gain by "snapping" its predictions to these levels, even when the features carry the information linearly. This can only enlarge the measured nonlinear advantage, so it makes the test conservative.

## 2. Question and hypotheses

Question: do tuned nonlinear probes read out the nominal fourth-order parameter α from frozen model features better than a tuned linear probe, by at least a tolerance Δ?

For each condition c: gap_c = R²(RBF kernel ridge) − R²(ridge). Both are cross-validated out-of-fold R², with identical standardization and nested tuning (`scripts/final_probe.py`, unchanged).

- H0_c: gap_c ≥ Δ
- H1_c: gap_c < Δ

The test is one-sided. A linear advantage (gap < 0) is reported but is not part of the hypothesis.

## 3. Tolerance

- **Δ = 0.01 R²**, i.e. 1% of the variance of α.
- **Sensitivity (reported regardless of outcome):** Δ = 0.005 and Δ = 0.02. They do not change the primary decision.
- **Context for the choice:**
  - between independent stimulus sets, the probe R² itself reproduced within 0.007, and the SD noise-seed SD of R² was ≤ 0.016 in the exploratory data;
  - 0.01 is about 1/60 of a 0.59 gap produced by a mismatched probe comparison on the same kind of features;
  - nonlinear advantages for a natural-texture fourth-order statistic were 0.006–0.045 (seven of ten ≥ 0.01).
  - Set-to-set variability of the gap itself was not measured before this test.

## 4. Confirmatory families

**Family P (primary, K = 8):**
- CLIP ViT-B/32, MAE ViT-B/16, DINO ViT-B/16, supervised ViT-B/16 (ImageNet-1k, timm `augreg_in1k`), and supervised ViT-B/16 (ImageNet-21k → 1k), each at k = 4 (56 × 56 check grid, 224-px input);
- ResNet50 at k = 4;
- ImageGPT-medium at its native 32 × 32 grid (k = 1);
- Stable Diffusion v1.5 UNet mid block at t = 500, k = 4 (128 × 128 grid, 512-px input).

**Family H (headroom, K = 6):** the six 224-px models at k = 8 (28 × 28 grid).

Each family is a separate claim at familywise error 0.05. Families P and H share the same stimulus sets, so their results are correlated; the Bonferroni bound is valid under any correlation.

## 5. Stimuli

- **Generator:** `stimuli/generate_btc_octave.py` at the frozen commit; GNU Octave with the BTC toolbox; α-only maximum-entropy textures; nonzero levels 0.05, 0.1, 0.2, 0.3, 0.4, 0.5; 200 maps per level. The α = 0 level is not generated.
- **Sets:** M_224 = 10 valid sets for the 224-px and ImageGPT conditions; M_SD = 5 valid sets for SD.
  - Set m = 0, 1, 2, … is written to `data/stimuli_conf/set{m}` with set name `c{m}` and `--base-seed 40000 + 100·m`. Seeds are assigned sequentially within a set (at most 24 per set).
  - Each set's grids are 28, 56, and 32. The first five valid sets also get 128, for SD.
  - Sets are generated in order of m until 10 valid sets exist (m ≤ 19, so seeds stay within 40000–41999).
  - The block 40000–41999 has not been used. Seeds already used: 10000–10020, 20000–20027, and 30000–30013. The dry run uses 49900–49923.
- **Validation:** the generator's own check (realized α within 5 SE of nominal; all first- to third-order statistics within 5 SE of 0).
  - If any file fails, set m is discarded as a whole and the next index is generated.
  - Every discarded set is reported.

## 6. Features and probes

- **Extraction:** exactly as in the main analysis. Exact integer replication (`--resize nearest`).
  - 224-px models: `scripts/extract_features_v2.py`.
  - ImageGPT: same script; native 32 × 32 grid.
  - SD: `scripts/extract_sd_seeded.py`; mid block, spatial mean pooling, null-text conditioning, pretrained VAE posterior mean; noise seeds 0–4 for every set; t = 500.
- **Driver:** `run_confirm.sh` runs generation, validation, extraction, probes, the confirmatory statistics, and the secondary analyses in that order.
- **Probes:** `scripts/final_probe.py` unchanged (ridge and RBF kernel ridge; standardization; nested tuning by shuffled, stratified 3-fold inner CV; 10 × stratified 5-fold outer CV, seed 0).
- **Execution host:** SD extraction may run on the workstation (RTX A4500) or on a DGX node.
  - Before any DGX output is used, the DGX re-extracts one existing condition (SD t = 500, g128, noise seed 0, main set). Its out-of-fold ridge R² must match the workstation's within 0.002.
  - The host of every file is recorded in the run log. With `RUN_SD=0`, the driver skips SD extraction; the SD probes and the statistics then run after the SD features are copied back.

## 7. Analysis (frozen in `scripts/r2_confirm_stats.py`)

1. **Per set m and condition c:** g_cm = mean over the 10 outer repeats of R²_kernel − R²_ridge. For SD, g_cm is the mean of this over noise seeds 0–4 (the seeds share the set's stimuli).
2. **Per condition:** ḡ_c = mean over sets; s_c = SD over sets; n_c = number of sets (M_224 or M_SD).
3. **Simultaneous one-sided 95% upper bound:** UB_c = ḡ_c + t_{1 − 0.05/K, n_c − 1} · s_c / √n_c (Bonferroni over the K conditions of the family).
4. **Decision rule:** the family claim "no condition has a nonlinear advantage of Δ or more" holds if UB_c < Δ for every c in the family. This is the only decision rule.
5. **Incomplete data:** the script refuses to compute if a condition has other than n_c sets, or an SD set lacks any of noise seeds 0–4. Missing outputs are rerun, not dropped.
6. **Reported for every condition:**
   - ḡ_c, s_c, and UB_c;
   - the marginal one-sided 95% bound;
   - the per-set g_cm;
   - the ridge and kernel R²;
   - the sensitivity decisions at Δ = 0.005 and Δ = 0.02.

## 8. Secondary analyses (declared; not part of the confirmatory claim)

- A tuned MLP (128–32) on Family P, first valid set only, one outer repeat, as in the main analysis.
- Empirical per-image α (check scale) as the target, and leave-one-α-level-out evaluation, on the first valid set (Families P and H). These analyses were specified after exploratory results on the earlier stimulus sets.

## 9. Dry run and deviations

- **Dry run:** one set with seeds 49900–49923 (set name `dry`, driver `run_confirm_dryrun.sh`), all conditions. It is used only to time the run and to check the pipeline end to end. Its data are exploratory and are not part of the test.
- **Sizing M (exploratory, before freezing):**
  - The between-set SD of the gap was estimated from the two existing sets (main, replication) that share seven Family P conditions: pooled SD ≈ 0.0004 (7 df). At M = 10 this gives a Bonferroni margin t · s / √M ≈ 0.0004; at M = 5, ≈ 0.0008.
  - No replication exists at k = 8, so Family H's between-set SD is unknown. The dry set adds one k = 8 set, which is not enough to estimate it; M_224 = 10 is chosen to leave margin.
- **After freezing:** any deviation (failed jobs, host change, software change) is recorded with its reason and reported. Freezing fixes the sets, families, Δ, and decision rule; none of them is changed after the first confirmatory stimulus is generated.

## 10. Reporting commitment

The outcome is reported in the manuscript whichever way it falls, together with the sensitivity results and every deviation.

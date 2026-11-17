# Wafer Map Defect Classifier: a CNN-style pipeline with no backprop

ES 335 (Machine Learning) project by Rishi Kumar and Satyam Suman.

## The question

**Can a one-layer, CNN-style pipeline whose filters are learned without backprop
(k-means on small patches) sort wafer maps into their defect patterns, and keep
working when a wafer comes in rotated or a little off-centre?**

- **Unit:** one wafer map, a grid of dies where 0 = off the wafer, 1 = passing die, 2 = failing die.
- **Target:** one of 9 labels: Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none.
- **What the model sees when it predicts:** only the wafer map itself. Lot name, die size and wafer
  index are never used as features. The lot name is used for one thing only: deciding which wafers
  go to training and which go to test.
- **Main score:** macro-F1, the plain average of the 9 per-class F1 scores. Accuracy would let the
  big classes (Edge-Ring, none) hide how badly the rare ones do.
- **Limits of the claim:** labeled wafers only, one public dataset, and a class mix with far more
  defects than a real fab (see [Limitations](#limitations)). We don't claim it would hold up on
  another fab's data without checking.

## The data

[WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) (`LSWMD.pkl`), real wafer maps
from Wu, Jang and Chen (2015), *IEEE Transactions on Semiconductor Manufacturing*. We don't
redistribute it: download it yourself and put it at `data/LSWMD.pkl` (it's gitignored).

What the audit (`python main.py audit`) found, and what we did about it:

| Finding | What we did |
|---|---|
| 811,457 wafers, but only 172,950 (21%) have a label | Use labeled wafers only |
| Of those, 147,431 are `none` (85%) and 25,519 have a defect | Keep every defect wafer + 10,000 random `none` = 35,519 wafers |
| Wafers come in lots (cassettes of up to 25) that share conditions | Split by lot, never by single wafer |
| Some identical maps show up in different lots | Treat lots that share an identical map as one group |
| 3 identical maps carry two different labels (e.g. Center vs Loc) | Kept as-is; a sign the labels have some noise |
| The built-in train/test column (spelled `trianTestLabel`) is lot-disjoint but very skewed (88% of Edge-Ring in train, 76% of none in test) | Not used; we made our own split |
| Map sizes range from 18 to 212 dies across (median 38) | Resize to 32x32 by picking the nearest die (no blending) |

## How we kept the test honest

- **Split:** 20% of lot groups are locked away as the test set (7,236 wafers, 2,051 lots). The other
  80% (28,283 wafers, 8,176 lots) get 5-fold cross-validation, grouped by lot.
  `python main.py check` re-checks that nothing leaks (21 checks, including one that proves the
  checker catches a leak when there is one).
- **Every learned step is fit on training folds only:** patch statistics, ZCA whitening, k-means
  filters, scaler, classifier, calibration.
- **All tuning happened on validation.** The final recipe was written to
  `experiments/final_config.json` and pushed to GitHub *before* the test set was loaded, so the
  commit timestamp shows it wasn't picked after peeking.
- **The test set was scored once** (`scripts/05_final_test.py` refuses to run twice). Two later looks
  are labelled post-hoc: the median-denoise comparison and the textbook CNN reference.
- **Error bars** come from redrawing whole test lots 1,000 times, since wafers from one lot look alike.

## The pipeline

1. **Clean up:** remove lonely failing dies (no failing neighbour among the 8 around it). This keeps
   1-die-thin scratches, which a median filter would erase. We keep 2 summary numbers so the
   Random class isn't wiped out: overall failing fraction and the share that cleanup removed.
2. **Resize** to 32x32 and split into a fail map and a wafer mask.
3. **Patches:** 200,000 on-wafer 5x5 patches from training wafers, contrast-normalised, then ZCA-whitened.
4. **Filters:** k-means with K = 32, with no backprop anywhere.
5. **Convolution:** slide every filter over the wafer, with a triangle activation. Each filter is
   tried at 0/90/180/270 degrees and the strongest response is kept.
6. **Ring pooling:** average the responses inside 4 rings (0-25-50-75-100% of the radius) plus the
   whole wafer, counting on-wafer dies only. Rings don't care which way the wafer is turned.
7. **Classifier:** logistic regression with balanced class weights and temperature scaling for honest
   probabilities. 32 filters x 5 regions + 2 summary numbers = 162 features.

## Results (locked test set, from `experiments/log.csv`)

| Model | Macro-F1 (95% lot bootstrap) | Accuracy |
|---|---|---|
| B0 majority class (validation) | 0.049 | 0.284 |
| B2 logistic on 1,024 flat pixels | 0.560 (0.535 to 0.584) | 0.682 |
| B2 random forest on flat pixels (best baseline) | 0.709 (0.688 to 0.726) | 0.794 |
| **Ours: k-means + logistic regression** | **0.765 (0.744 to 0.783)** | **0.820** |
| Ours with linear SVM instead | 0.759 (0.737 to 0.778) | 0.826 |
| Reference: textbook CNN with backprop (post-hoc) | 0.829 (0.812 to 0.843) | 0.884 |

- We beat the best baseline by +0.056 (+0.038 to +0.074, same redrawn lots).
- Our probabilities are honest: calibration error (ECE) 0.025, against 0.240 for the random forest.
- **Rotation:** at 90/180/270 degrees we score 0.764 to 0.767, the same as unrotated. The baselines
  and the CNN all lose points. In-between angles dip to about 0.67 at 45 degrees.
- **Shift:** sliding a wafer 3 pixels costs us 0.08 (0.683). Flat pixels lose 0.28 to 0.41, the CNN 0.20.
- **Data needs:** on validation, 5% of the training lots already beats flat pixels trained on 100%.
- **The CNN wins on score.** A 3-layer textbook CNN beats us by 0.047 to 0.081 macro-F1. We get
  about 92% of its macro-F1 with 48x fewer learned numbers (3,242 vs 155,801), no backprop, and
  steadier behaviour under rotation and shift.
- **Denoise trade-off (post-hoc):** median denoising would score +0.033 higher (+0.016 to +0.051),
  but it catches fewer scratches: Scratch recall falls from 0.711 to 0.596. Which one is better
  depends on whether a fab cares most about catching scratches.

Figures are in `experiments/figures/`, and every number above is a row in `experiments/log.csv`
or a field in the JSON files next to it.

## Limitations

- **Our test set has far more defects than a real fab.** It is 27% `none`; among labeled real wafers
  `none` is 85%. Recall doesn't depend on that, but precision does. Replaying our test error rates
  at real class shares (`python main.py real-world-precision`) drops macro-F1 from 0.765 to 0.646,
  and Scratch precision from 0.267 to 0.055, so only about 1 in 18 Scratch alarms would be a real
  scratch. 19% of clean wafers get flagged as some defect. That's fine for a first-pass filter in
  front of a human, but not for full automation.
- **Scratch and Loc are our weak spots** (test F1 0.388 and 0.560). Short scratches and small blobs
  look alike through 5x5 windows. The two classes are each other's most common mistake.
- **Rotation is nearly exact at quarter turns, not perfectly exact.** The ZCA step learns an average
  patch that isn't rotation-symmetric, so a 90 degree turn moves the features slightly
  (macro-F1 moves by about 0.002).
- **The learned filters are not what makes it work.** Random 5x5 filters, on the same validation
  folds and the same pipeline, beat our k-means filters by +0.015 macro-F1 (+0.007 to +0.024 over
  1,000 lot redraws, and better in all 5 folds; `python main.py filter-check`). So the strength comes
  from the *structure*: sliding small filters, trying 4 orientations, and pooling in rings. The
  k-means filters still earn their place by being readable (spots, short lines, edges you can show
  on the website), but we don't claim they're more accurate. We kept k-means because the recipe was
  frozen before the test and we only score the test once.
- **Labels have some noise** (3 identical maps with conflicting labels), and 79% of WM-811K has no
  label at all, so we can't say how the model behaves on the unlabeled majority.
- **The learning curve's 1% point** has no Near-full wafers in its samples, so it understates that size.

## Run it yourself

Python 3.13. Versions are pinned so everyone gets the same numbers.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put `LSWMD.pkl` in `data/`, then run these in order:

| Command | What it does | Time |
|---|---|---|
| `python main.py verify` | Checks the dataset loads with the expected row counts | ~1 min |
| `python main.py audit` | Data audit, builds the 35,519-wafer subset | ~1 min |
| `python main.py splits` | Makes and freezes the lot-grouped split | ~1 min |
| `python main.py check` | 21 leak checks on the split | seconds |
| `python main.py baselines` | B0 to B3 on validation | ~2 min |
| `python main.py pipeline` | 3 filter types x 4 classifiers on validation | ~17 min |
| `python main.py tuning` | 7 one-change ablations on validation | ~40 min |
| `python main.py filter-check` | Paired k-means vs random filters check on validation | ~10 min |
| `python main.py final-test` | **The one test run.** Refuses to run twice | ~2.5 min |
| `python main.py analysis` | Rotation, shift, learning curve, errors, filters | ~32 min |
| `python main.py posthoc-median` | Post-hoc median vs isolated denoise on test | ~1.5 min |
| `python main.py reference-cnn` | Textbook CNN, epochs picked on validation, test once | ~25 min |
| `python main.py real-world-precision` | Precision replayed at real class shares | seconds |

`final-test` and `reference-cnn` each refuse to run once their results file exists. To reproduce
from scratch, delete `experiments/final_test.json` and `experiments/reference_cnn.json` first.
Fitted models are saved to `data/` (gitignored) and come back when you rerun those two commands.

Seeds are fixed everywhere, but k-means and the linear algebra use several CPU threads, so a
rerun of the Phase 3 scripts can move a macro-F1 by up to about 0.001. That's far smaller than the
fold-to-fold spread logged next to every result (0.004 to 0.014).

## Repo map

```
main.py                  one entry point for every step
src/                     data loading, preprocessing, conv features, models, evaluation helpers
scripts/                 one numbered script per step (01_audit ... 10_real_world_precision)
splits/                  the frozen split and its summary
experiments/log.csv      every logged result; JSON/CSV files and figures sit next to it
```

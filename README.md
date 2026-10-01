# Diabetes Risk Check

Free, private, in-browser pattern check for early-diabetes symptoms. Answer
16 dropdown questions; a 5-expert stacking ensemble scores your pattern
**locally in your browser** — no server inference, nothing uploaded.

A score is a prompt to get a blood test, never a diagnosis.

## Run locally

```
python3 -m http.server 8900
# open http://localhost:8900
```

(`cleanUrls` in vercel.json maps /how etc. under Vercel; locally use
/how.html if /how 404s.)

## Retrain from scratch

```
python3 -m venv diabetes-venv
diabetes-venv/bin/pip install -r train/requirements.txt
diabetes-venv/bin/python train/compare.py   # 10 algos + ensemble -> train/comparison.json
diabetes-venv/bin/python train/export.py    # model + parity fixture -> train/model.json
diabetes-venv/bin/python train/gan_experiment.py  # CTGAN study -> train/gan_results.json
node train/parity.js                        # JS must equal the shipped artifact
cp train/model.json train/comparison.json train/gan_results.json site/public/
diabetes-venv/bin/python train/build.py     # hashed bundle -> deploy/
bash train/verify_all.sh                    # full pre-deploy gate
```

Copy the fresh `deploy/` contents to the repo root to publish (the root is
what Vercel serves).

## Structure

- `site/public/` — source: index (assessment), how, summary, about, gan
- `site/vercel.json` — source-of-truth headers for unhashed local deploys
- `train/` — compare, export, parity (+fixture), gan experiment, build, gates
- `diabetes-repo/data.csv` — training questionnaire (UCI, see below)
- repo root — built bundle (hashed assets + per-page CSP), what gets deployed

Generated JSON (`train/*.json`, `site/public/*.json`) is git-ignored; the
root `comparison.json` / `model.*.json` / `gan_results.*.json` are the served
copies and stay tracked.

## Methodology

- **Duplicates dropped first.** The CSV has 520 rows but 251 unique feature
  profiles, with no conflicting labels. Everything — comparison, export,
  GAN study — runs on unique profiles, so no test row is a copy of a
  training row.
- **Leakage-safe CV.** Scaling is fit inside every fold; repeated stratified
  5×4 out-of-fold probabilities; bootstrap CIs computed over the same vector
  the point estimate used (hard labels for accuracy, probabilities for AUC —
  the build asserts each interval covers its point).
- **Shipped stack, honestly described.** The browser model's stacker is fit
  on out-of-fold expert probabilities; experts are refit once on all unique
  rows. `comparison.json` numbers describe the recipe, not a test of the
  exported weights — the summary page says exactly that, and names the
  per-metric winners from the data instead of hard-coding them.
- **Parity against the artifact.** `train/export.py` writes
  `parity_fixture.json` from the fitted objects; `parity.js` checks
  `model.js` against that fixture — never against a fresh retrain.
- **Hash-after-rewrite.** Entry scripts are repointed at hashed data names
  before being hashed themselves, so immutable filenames always match the
  served bytes (asserted by the build).
- **No silent defaults.** Age and sex are required; a blank form cannot be
  scored. No-symptom answers get an on-card caveat, not a high-risk alarm.

## Dataset

UCI Early Stage Diabetes Risk Prediction — questionnaire answers collected
at Sylhet Diabetes Hospital, Bangladesh (Islam et al. 2019;
DOI 10.24432/C5VG8H). 520 rows, 16 symptom/sex/age fields, 61.5% positive
on rows. This is a clinic questionnaire, not population sampling — the site
calls the output a pattern match for that reason.

This tool provides educational estimates only and is not medical advice.

# Diabetes Risk Check

Free, private, in-browser early-diabetes screening. Answer 16 dropdown
questions; a 5-expert stacking ensemble (trained with leakage-safe repeated
cross-validation on the 520-record Kaggle/NIDDK dataset) scores your pattern
**locally in your browser** — no server inference, nothing uploaded.

**Live:** deploy `diabetes-site/` on Vercel (drag the folder at vercel.com/new
or `vercel` from inside it).

## Run locally
```
python3 -m http.server 8900 --directory diabetes-site
# open http://localhost:8900
```
(`cleanUrls` in vercel.json maps /how etc. under Vercel; locally use
/how.html if /how 404s.)

## Retrain from scratch
```
python3 -m venv diabetes-venv
diabetes-venv/bin/pip install -r train/requirements.txt
diabetes-venv/bin/python train/compare.py   # 10 algos + ensemble -> train/comparison.json
diabetes-venv/bin/python train/export.py    # export -> train/model.json
node train/parity.js                        # JS==sklearn parity (all gates)
cp train/model.json train/comparison.json site/public/
diabetes-venv/bin/python train/build.py     # hashed bundle -> deploy/
bash train/verify_all.sh                    # full pre-deploy gate
```

## Structure
- `site/public/` — source: index (assessment), how, summary, about
- `train/` — comparison, export, parity, build pipeline, mobile audit
- `deploy/` — built bundle (hashed assets + per-page CSP)
- `diabetes-site/` — deployable copy of the built bundle

## Methodology
Leakage-safe: scaling fit inside every CV fold; repeated stratified OOF for
every metric; bootstrap CIs; stacking CV=5. Bands and caveats derived from
measured score distribution. Known dataset artifact (9 zero-symptom records,
6/9 positive, all female) is disclosed in-app rather than fudged.

This tool provides educational risk estimates only and is not medical advice.

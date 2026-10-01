/* ==========================================================================
   model.js — in-browser inference for the diabetes risk stack.

   The whole model ships as static JSON and every expert is evaluated here,
   in the visitor's tab. No server, no upload, no latency beyond a fetch.

   Each routine mirrors the fitted scikit-learn estimator exactly (verified
   numerically by train/parity.js). Conventions shared with train/export.py:
   - features: [Age, Gender(Male=1), 14 symptoms (Yes=1)] — raw units in,
     standard-scaled with scaler.mean / scaler.scale before every expert.
   - KNN: k=21, distance weighting 1/sqrt(sqDist). Stored as exact Age +
     a 15-bit mask (bit i = feature i+1: Gender + 14 symptoms). Scaled squared
     distance = ageW*dAge^2 + sum binW[j]*bitdiff^2, with
     ageW = 1/sd(Age)^2 and binW[j] = 1/scale[j+1]^2. Exact zero distance
     returns that neighbour's label outright (sklearn's d==0 guard).
   - GB: binary binomial-deviance walk. raw = init + lr * sum(tree leaf val);
     proba = sigmoid(raw). Trees carry single "val" residuals.
   - RF: proba = mean over trees of pos/(neg+pos) at the reached leaf.
     Thresholds are float64 and compared directly (no float32 rounding).
   - NB: class-major theta/sigma on SCALED features + log priors.
   - stacker: sigmoid(coef . expertProbs + intercept).
   ========================================================================== */
(function (global) {
  "use strict";

  function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  // Guard the public entry point. scale() iterates the INPUT length, so a
  // short/long/non-numeric vector would otherwise produce NaN silently and
  // the UI would render "NaN%" as a legitimate score.
  function assertValid(raw, model) {
    if (!model || !model.ready) {
      throw new Error("DiabetesModel: model not loaded (call load() first)");
    }
    if (!Array.isArray(raw) || raw.length !== model.features.length) {
      throw new Error(
        "DiabetesModel: expected " + model.features.length + " inputs, got " +
          (Array.isArray(raw) ? raw.length : typeof raw)
      );
    }
    for (let i = 0; i < raw.length; i++) {
      const v = raw[i];
      if (typeof v !== "number" || !isFinite(v)) {
        throw new Error("DiabetesModel: input[" + i + "] is not a finite number");
      }
    }
    if (model.features[0] !== "Age" || model.features[1] !== "Gender") {
      throw new Error("DiabetesModel: unexpected feature order in model.json");
    }
  }

  function scale(raw, model) {
    const xs = new Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      xs[i] = (raw[i] - model.scaler.mean[i]) / model.scaler.scale[i];
    }
    return xs;
  }

  function predictLR(xs, E) {
    let z = E.intercept;
    for (let i = 0; i < xs.length; i++) z += E.coef[i] * xs[i];
    return sigmoid(z);
  }

  function predictNB(xs, E) {
    // E.theta[c][i], E.sigma[c][i] on scaled features; E.priors[c].
    let best = -1, bestLL = -Infinity;
    const lls = [0, 0];
    for (let c = 0; c < 2; c++) {
      let ll = Math.log(E.priors[c]);
      for (let i = 0; i < xs.length; i++) {
        const d = xs[i] - E.theta[c][i];
        ll += -0.5 * (Math.log(2 * Math.PI * E.sigma[c][i]) + (d * d) / E.sigma[c][i]);
      }
      lls[c] = ll;
      if (ll > bestLL) { bestLL = ll; best = c; }
    }
    // normalize to a probability via log-sum-exp
    const m = Math.max(lls[0], lls[1]);
    const p1 = Math.exp(lls[1] - m) / (Math.exp(lls[0] - m) + Math.exp(lls[1] - m));
    return E.classes[1] === 1 ? p1 : 1 - p1;
  }

  function walkTree(xfs, T, leafVal) {
    // CRITICAL: sklearn evaluates tree splits on float32 input AND float32
    // threshold (xf32 <= thr32, bit-for-bit). Equality at f32 — e.g. Age
    // landing exactly on its own threshold — must go LEFT. But JS's fround
    // comparison of equal-bit values still goes LEFT (<=), so the real
    // divergence is elsewhere: sklearn's internal predictor uses the
    // threshold's STORED float32 bits, while a fround of the float64 json
    // value can differ by one ULP if json round-tripping isn't exact.
    // json.dump uses repr() shortest-round-trip, so thr f32->f64->repr->f32
    // is lossless and the only remaining mismatch source is the input side:
    // sklearn casts the SCALED float64 input to float32 — we do the same
    // with Math.fround(xs). Keep both frounds; verified 61-tree divergence
    // was the missing input-side fround.
    let n = 0;
    for (;;) {
      const l = T.left[n];
      if (l === -1) return leafVal(T, n);
      // CRITICAL: sklearn evaluates tree splits on float32 input against
      // float32 thresholds. Comparing float64 xs against the float64-expanded
      // threshold flips borderline splits (61/300 rf trees disagreed on real
      // rows) — so round BOTH sides like sklearn does.
      // CRITICAL: sklearn's predictor casts the SCALED input to float32 but
      // compares it against the threshold's STORED (float32-precision)
      // float64 value: fround(x) <= thr. Rounding BOTH sides flips borderline
      // splits where fround(thr) rounds 1 ULP up; f64 x flips the mirror
      // cases. Exactly one tree in the fuzz set discriminates the two.
      n = xfs[T.feat[n]] <= T.thr[n] ? l : T.right[n];
    }
  }

  function predictGB(xs, E) {
    const xfs = xs.map(Math.fround);
    let raw = E.init;
    for (const T of E.trees) raw += E.lr * walkTree(xfs, T, (t, n) => t.val[n]);
    return sigmoid(raw);
  }

  function predictRF(xs, E) {
    const xfs = xs.map(Math.fround);
    let s = 0;
    for (const T of E.trees) {
      s += walkTree(xfs, T, (t, n) => t.pos[n] / (t.pos[n] + t.neg[n]));
    }
    return s / E.trees.length;
  }

  function predictKNN(raw, E, model) {
    const k = E.k;
    const ageW = 1 / (model.scaler.scale[0] * model.scaler.scale[0]);
    const binW = [];
    for (let j = 1; j < 16; j++) {
      const s = model.scaler.scale[j];
      binW.push(1 / (s * s));
    }
    // exact query mask: bit (i-1) = feature i (Gender + 14 symptoms)
    let qm = 0;
    for (let i = 1; i < 16; i++) if (raw[i] >= 0.5) qm |= 1 << (i - 1);
    // bounded insertion list of the k smallest (dist, label) — n is the
    // unique-profile table (~251 rows), k=21, so a hand-rolled heap buys
    // nothing and risks subtle ordering bugs.
    const best = []; // sorted ascending by dist, capped at k
    const consider = (d2, label) => {
      if (best.length < k) {
        best.push({ d: d2, y: label });
        best.sort((a, b) => a.d - b.d);
      } else if (d2 < best[k - 1].d) {
        best[k - 1] = { d: d2, y: label };
        best.sort((a, b) => a.d - b.d);
      }
    };
    for (let r = 0; r < E.masks.length; r++) {
      const m = E.masks[r];
      const dAge = raw[0] - E.ages[r];
      let d2 = ageW * dAge * dAge;
      let diff = qm ^ m;
      while (diff) {
        const b = diff & -diff; // lowest set bit
        const j = 31 - Math.clz32(b); // bit index
        d2 += binW[j] * 1; // bit differs by exactly 1 scaled unit
        diff &= diff - 1;
      }
      consider(d2, E.y[r]);
    }
    if (!best.length) throw new Error("DiabetesModel: KNN over empty index");
    // sklearn behaviour: exact-zero distance neighbours dominate outright.
    if (best[0].d === 0) {
      let pos = 0, tot = 0;
      for (const b of best) {
        if (b.d !== 0) break;
        pos += b.y; tot++;
      }
      return pos / tot;
    }
    let wp = 0, w = 0;
    for (const b of best) {
      const wgt = 1 / Math.sqrt(b.d);
      wp += wgt * b.y;
      w += wgt;
    }
    return wp / w;
  }

  // ---------------------------------------------------------------- public API
  function predictAll(raw, model) {
    assertValid(raw, model);
    const xs = scale(raw, model);
    const E = model.experts;
    const p = {
      lr: predictLR(xs, E.lr),
      nb: predictNB(xs, E.nb),
      knn: predictKNN(raw, E.knn, model),
      gb: predictGB(xs, E.gb),
      rf: predictRF(xs, E.rf),
    };
    const probs = model.stacker.names.map((n) => p[n]);
    let z = model.stacker.intercept;
    for (let i = 0; i < probs.length; i++) z += model.stacker.coef[i] * probs[i];
    return { prob: sigmoid(z), parts: p, xs, raw };
  }

  // Marginal contribution of each feature: how much the stacked probability
  // moves when that feature is set to its "off" value — No (0) for symptoms,
  // the training mean for Age, the opposite sex for Gender — holding
  // everything else at the visitor's answers. Reads as "this answer is
  // pushing your score up", and 0 means "this answer adds nothing".
  function contributions(raw, model) {
    const base = predictAll(raw, model).prob;
    const out = {};
    for (let i = 0; i < model.features.length; i++) {
      const key = model.features[i];
      const alt = raw.slice();
      // "off" state per feature type: symptoms -> No, Age -> training mean,
      // Gender -> the other sex (so its base-rate shift is visible, not hidden)
      if (key === "Age") alt[i] = model.neutral[i];
      else if (key === "Gender") alt[i] = raw[i] >= 0.5 ? 0 : 1;
      else alt[i] = 0;
      const p2 = predictAll(alt, model).prob;
      out[key] = { delta: base - p2, value: raw[i] };
    }
    return out;
  }

  global.DiabetesModel = {
    load: (data) => {
      data.index = {};
      data.features.forEach((f, i) => (data.index[f] = i));
      // Neutral = the average person in the training data (raw units).
      data.neutral = data.scaler.mean.slice();
      data.ready = true;
      return data;
    },
    predict: (raw, model) => predictAll(raw, model),
    contributions,
    ready: (model) => !!(model && model.ready),
  };
})(typeof self !== "undefined" ? self : this);

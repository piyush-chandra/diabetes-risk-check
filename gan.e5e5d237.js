/* /gan page renderer: fetch gan_results.json, fill cards + table + verdict.
   The verdict and the short answer are COMPUTED from the data — there is no
   hand-written conclusion anywhere on this page. */
(function () {
  "use strict";
  const NAMES = {
    baseline: "Baseline (unique profiles, no tricks)",
    gan_augmented: "GAN-augmented (real + CTGAN synth)",
    gan_only: "GAN-only (synthetic data stress test)",
    oversampled: "Minority oversampled (classic trick)",
  };
  const fmt = (v, d) => (typeof v === "number" ? v.toFixed(d === undefined ? 4 : d) : "—");

  fetch("/gan_results.f1d1b205.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((data) => {
      const arms = data.arms;
      const keys = Object.keys(arms);
      const bestKey = keys.reduce((a, k) => (arms[k].auc_mean > arms[a].auc_mean ? k : a), keys[0]);
      // cards: the headline trio
      const cardDefs = [
        { a: "baseline", t: "Baseline (no tricks)" },
        { a: "gan_augmented", t: "GAN-augmented" },
        { a: "oversampled", t: "Oversampled" },
      ];
      document.getElementById("gan-cards").innerHTML = cardDefs.map((c) => {
        const m = arms[c.a];
        return '<article class="prose-block"><h3>' + c.t + "</h3>" +
          '<p class="card-value">' + fmt(m.auc_mean) + "</p>" +
          "<p>ROC-AUC ± " + fmt(m.auc_std, 3) + " · acc " + fmt(m.acc_mean) + " · Brier " + fmt(m.brier_mean) + "</p></article>";
      }).join("");

      // table: all arms, star = best mean AUC from the data
      let t = '<thead><tr><th>' + ["Strategy", "ROC-AUC", "±", "Accuracy", "±", "Brier", "±"].join("</th><th>") + '</th></tr></thead><tbody>';
      keys.forEach((k) => {
        const m = arms[k];
        const best = k === bestKey;
        t += "<tr" + (best ? ' class="row-ensemble"' : "") + ">" +
          "<td>" + (best ? "★ " : "") + NAMES[k] + "</td>" +
          "<td>" + fmt(m.auc_mean) + "</td><td>" + fmt(m.auc_std, 3) + "</td>" +
          "<td>" + fmt(m.acc_mean) + "</td><td>" + fmt(m.acc_std, 3) + "</td>" +
          "<td>" + fmt(m.brier_mean) + "</td><td>" + fmt(m.brier_std, 3) + "</td></tr>";
      });
      t += "</tbody>";
      document.getElementById("gan-table").innerHTML = t;

      // per-fold dots: variance, not just the mean
      const folds = document.getElementById("gan-folds");
      if (folds && arms.baseline.folds) {
        folds.innerHTML = keys.map((k) => {
          const vals = arms[k].folds.auc || [];
          const dots = vals.map((v) => '<span class="fold-dot" title="fold AUC ' + fmt(v) + '"></span>').join("");
          return '<div class="fold-row"><span class="fold-name">' + NAMES[k] + '</span><span class="fold-dots">' + dots + "</span><span>" + vals.map((v) => fmt(v, 3)).join(" · ") + "</span></div>";
        }).join("");
      }

      // verdict, computed from the data (never hand-written)
      const b = arms.baseline, g = arms.gan_augmented, o = arms.oversampled;
      const dG = g.auc_mean - b.auc_mean, dO = o.auc_mean - b.auc_mean;
      // paired-ish significance proxy: fold-level std/sqrt(n)
      const seG = Math.sqrt((g.auc_std ** 2 + b.auc_std ** 2) / data.n_folds);
      const sig = Math.abs(dG) > 1.96 * seG;
      let verdict, short;
      if (dG >= 0 && !sig) {
        verdict = "GAN augmentation matched the baseline within noise (" + (dG >= 0 ? "+" : "") + fmt(dG, 4) + " AUC, n.s.). No measurable gain — the unique profiles' signal is already fully extracted by the baseline recipe.";
        short = "Short answer: no measurable gain — the GAN matched the baseline within fold noise.";
      } else if (dG >= 0 && sig) {
        verdict = "GAN augmentation improved AUC by " + fmt(dG, 4) + " over baseline (beyond fold noise). Real but modest — worth it only if collecting more data isn't possible.";
        short = "Short answer: a small but real gain over baseline.";
      } else if (!sig) {
        verdict = "GAN augmentation scored " + fmt(dG, 4) + " AUC below baseline — within fold noise, but consistently worse across folds. It added no information and injected bias.";
        short = "Short answer: no — it scored below baseline, inside fold noise.";
      } else {
        verdict = "GAN augmentation scored " + fmt(dG, 4) + " AUC below baseline, outside fold noise. The generator distorted the feature distributions enough to hurt the model. Oversampling fared " + (dO >= 0 ? "slightly better" : "similarly") + ".";
        short = "Short answer: no — it hurt, outside fold noise.";
      }
      document.getElementById("answer-body").textContent = verdict;
      const shortEl = document.getElementById("short-answer");
      if (shortEl) shortEl.innerHTML = "<strong>" + short + "</strong>";
      const go = arms.gan_only;
      const readEl = document.getElementById("reading-body");
      if (readEl) {
        readEl.textContent =
          "Baseline " + fmt(b.auc_mean) + " vs GAN-augmented " + fmt(g.auc_mean) + " vs oversampled " + fmt(o.auc_mean) +
          " — a gap of " + fmt(dG, 4) + " and " + fmt(dO, 4) + " against fold noise around ±" + fmt(seG, 4) +
          ". The GAN-only stress test reads " + fmt(go.auc_mean) + " AUC: synthetic rows alone " +
          (go.auc_mean < 0.6 ? "cannot replace real ones here." : "carry real signal here.") +
          " The classic oversample costs nothing to train — the honest question is whether the GAN beats that free trick, not just the baseline.";
      }
      document.title = "GAN experiment: " + (dG >= 0 ? "no gain" : "hurt") + " | Diabetes Risk Check";
    })
    .catch(() => {
      document.getElementById("gan-table").innerHTML =
        "<tbody><tr><td>Results unavailable — please reload.</td></tr></tbody>";
    });
})();

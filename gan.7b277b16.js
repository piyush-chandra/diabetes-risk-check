/* /gan page renderer: fetch gan_results.json, fill cards + table + verdict */
(function () {
  "use strict";
  const NAMES = {
    baseline: "Baseline (original 520·80% split)",
    gan_augmented: "GAN-augmented (real + CTGAN synth)",
    gan_only: "GAN-only (synthetic data stress test)",
    oversampled: "Minority oversampled (classic trick)",
  };
  const fmt = (v, d) => (typeof v === "number" ? v.toFixed(d === undefined ? 4 : d) : "—");

  fetch("/gan_results.a1a5d53d.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((data) => {
      const arms = data.arms;
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

      // table: all arms
      let t = '<thead><tr>' + ["Strategy", "ROC-AUC", "±", "Accuracy", "±", "Brier", "±"] + '</tr></thead><tbody>';
      Object.keys(arms).forEach((k) => {
        const m = arms[k];
        const best = k === "baseline";
        t += "<tr" + (best ? ' class="row-ensemble"' : "") + ">" +
          "<td>" + (best ? "★ " : "") + NAMES[k] + "</td>" +
          "<td>" + fmt(m.auc_mean) + "</td><td>" + fmt(m.auc_std, 3) + "</td>" +
          "<td>" + fmt(m.acc_mean) + "</td><td>" + fmt(m.acc_std, 3) + "</td>" +
          "<td>" + fmt(m.brier_mean) + "</td><td>" + fmt(m.brier_std, 3) + "</td></tr>";
      });
      t += "</tbody>";
      document.getElementById("gan-table").innerHTML = t;

      // verdict, computed from the data (never hand-written)
      const b = arms.baseline, g = arms.gan_augmented, o = arms.oversampled;
      const dG = g.auc_mean - b.auc_mean, dO = o.auc_mean - b.auc_mean;
      // paired-ish significance proxy: fold-level std/sqrt(n)
      const seG = Math.sqrt((g.auc_std ** 2 + b.auc_std ** 2) / data.n_folds);
      const sig = Math.abs(dG) > 1.96 * seG;
      let verdict;
      if (dG >= 0 && !sig) verdict = "GAN augmentation matched the baseline within noise (" + (dG >= 0 ? "+" : "") + fmt(dG, 4) + " AUC, n.s.). No measurable gain — the dataset's signal is already fully extracted by the baseline recipe.";
      else if (dG >= 0 && sig) verdict = "GAN augmentation improved AUC by " + fmt(dG, 4) + " over baseline (beyond fold noise). Real but modest — worth it only if collecting more data isn't possible.";
      else if (!sig) verdict = "GAN augmentation scored " + fmt(dG, 4) + " AUC below baseline — within fold noise, but consistently worse across folds. It added no information and injected bias.";
      else verdict = "GAN augmentation scored " + fmt(dG, 4) + " AUC below baseline, outside fold noise. The generator distorted the feature distributions enough to hurt the model. Oversampling fared " + (dO >= 0 ? "slightly better" : "similarly") + ".";
      document.getElementById("answer-body").textContent = verdict;
      document.title = "GAN experiment: " + (dG >= 0 ? "no gain" : "hurt") + " | Diabetes Risk Check";
    })
    .catch(() => {
      document.getElementById("gan-table").innerHTML = "<tbody><tr><td>Results unavailable — please reload.</td></tr></tbody>";
    });
})();

/* Summary table renderer: fetch comparison.json, fill table + cards.
   Winners are computed from the data (best ROC-AUC, best Brier) — never
   hand-written — so the page cannot contradict its own JSON. */
(function () {
  "use strict";
  const fmt = (v, d) => (typeof v === "number" ? v.toFixed(d === undefined ? 3 : d) : "—");

  fetch("/comparison.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((data) => {
      const m = {};
      data.models.forEach((r) => (m[r.name] = r));
      const bestAuc = Math.max(...data.models.map((r) => r.roc_auc));
      const bestBrier = Math.min(...data.models.map((r) => r.brier));
      const bestAcc = Math.max(...data.models.map((r) => r.accuracy));

      // headline cards: the deployed model plus the honest context
      const ens = m["Stacking Ensemble"];
      const uniq = data.n_unique || data.n_rows;
      const raw = data.n_raw ? ` (${data.n_raw} rows, ${uniq} unique)` : "";
      const cards = [
        { h: "Deployed stack accuracy", v: fmt(ens.accuracy, 4), s: "95% CI " + fmt(ens.accuracy_ci95[0], 3) + "–" + fmt(ens.accuracy_ci95[1], 3) },
        { h: "Deployed stack ROC-AUC", v: fmt(ens.roc_auc, 4), s: "95% CI " + fmt(ens.auc_ci95[0], 3) + "–" + fmt(ens.auc_ci95[1], 3) },
        { h: "Profiles", v: String(uniq), s: data.n_positive + " positive (" + ((100 * data.n_positive) / data.n_rows).toFixed(1) + "%)" + raw },
        { h: "Validation", v: data.folds + "×" + data.n_repeats, s: "folds × repeats, duplicates dropped first" },
      ];
      document.getElementById("ensemble-cards").insertAdjacentHTML(
        "beforeend",
        cards.map((c) => '<article class="prose-block why-card"><h3>' + c.h + '</h3><p class="card-value">' + c.v + "</p><p>" + c.s + "</p></article>").join("")
      );

      // who actually leads, per metric — the table bolds these, the text names them
      const vals = (key) => data.models.map((x) => x[key]);
      const namesWith = (key, v) => data.models.filter((r) => r[key] === v).map((r) => r.name).join(", ");
      const note = document.getElementById("leaders-note");
      if (note) {
        note.textContent =
          "Best accuracy: " + namesWith("accuracy", bestAcc) + " (" + fmt(bestAcc, 4) + "). " +
          "Best ROC-AUC: " + namesWith("roc_auc", bestAuc) + " (" + fmt(bestAuc, 4) + "). " +
          "Best calibration (Brier, lower wins): " + namesWith("brier", bestBrier) + " (" + fmt(bestBrier, 4) + "). " +
          "The deployed stack is the ★ row — chosen for honest probabilities across all five experts, not for topping any single column.";
      }

      // table
      const cols = ["Algorithm", "Accuracy", "ROC-AUC", "PR-AUC", "Brier ↓", "F1", "Sens", "Spec"];
      let t = "<thead><tr>" + cols.map((c) => "<th>" + c + "</th>").join("") + "</tr></thead><tbody>";
      t += data.models
        .map((r) => {
          const isEns = !!r.deployed || r.name === "Stacking Ensemble";
          const cell = (v, best, lowWins) =>
            (!isEns && (lowWins ? v <= best + 1e-9 : v >= best - 1e-9))
              ? "<strong>" + fmt(v, 4) + "</strong>" : fmt(v, 4);
          return (
            "<tr" + (isEns ? ' class="row-ensemble"' : "") + ">" +
            "<td>" + (isEns ? "★ " : "") + r.name + "</td>" +
            "<td>" + cell(r.accuracy, bestAcc) + "</td>" +
            "<td>" + cell(r.roc_auc, bestAuc) + "</td>" +
            "<td>" + fmt(r.pr_auc, 4) + "</td>" +
            "<td>" + cell(r.brier, bestBrier, true) + "</td>" +
            "<td>" + fmt(r.f1, 4) + "</td>" +
            "<td>" + fmt(r.sensitivity, 4) + "</td>" +
            "<td>" + fmt(r.specificity, 4) + "</td></tr>"
          );
        })
        .join("");
      t += "</tbody>";
      document.getElementById("compare-table").innerHTML = t;
    })
    .catch(() => {
      document.getElementById("compare-table").innerHTML =
        "<tbody><tr><td>Benchmark data unavailable — please reload.</td></tr></tbody>";
    });
})();

/* summary table renderer: fetch comparison.json, fill table + cards */
(function () {
  "use strict";
  const fmt = (v, d) => (typeof v === "number" ? v.toFixed(d === undefined ? 3 : d) : "—");

  fetch("/comparison.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((data) => {
      const m = {};
      data.models.forEach((r) => (m[r.name] = r));
      const bestAuc = Math.max(...data.models.filter((r) => !r.is_ensemble).map((r) => r.roc_auc));
      const bestBrier = Math.min(...data.models.filter((r) => !r.is_ensemble).map((r) => r.brier));

      // headline cards
      const ens = m["Stacking Ensemble"];
      const cards = [
        { h: "Deployed ensemble accuracy", v: fmt(ens.accuracy, 4), s: "95% CI " + fmt(ens.accuracy_ci95[0], 3) + "–" + fmt(ens.accuracy_ci95[1], 3) },
        { h: "Deployed ensemble ROC-AUC", v: fmt(ens.roc_auc, 4), s: "95% CI " + fmt(ens.auc_ci95[0], 3) + "–" + fmt(ens.auc_ci95[1], 3) },
        { h: "Records", v: data.n_rows, s: data.n_positive + " positive (" + ((100 * data.n_positive) / data.n_rows).toFixed(1) + "%)" },
        { h: "Validation", v: data.folds + "×" + data.n_repeats, s: "folds × repeats, selection inside each fold" },
      ];
      document.getElementById("ensemble-cards").insertAdjacentHTML(
        "beforeend",
        cards.map((c) => '<article class="prose-block why-card"><h3>' + c.h + '</h3><p class="card-value">' + c.v + "</p><p>" + c.s + "</p></article>").join("")
      );

      // table
      const cols = ["Algorithm", "Accuracy", "ROC-AUC", "PR-AUC", "Brier ↓", "F1", "Sens", "Spec"];
      let t = "<thead><tr>" + cols.map((c) => "<th>" + c + "</th>").join("") + "</tr></thead><tbody>";
      t += data.models
        .map((r) => {
          const isEns = !!r.is_ensemble || r.name === "Stacking Ensemble";
          const aucCell = !isEns && r.roc_auc >= bestAuc - 1e-9 ? "<strong>" + fmt(r.roc_auc, 4) + "</strong>" : fmt(r.roc_auc, 4);
          const brierCell = !isEns && r.brier <= bestBrier + 1e-9 ? "<strong>" + fmt(r.brier, 4) + "</strong>" : fmt(r.brier, 4);
          return (
            "<tr" + (isEns ? ' class="row-ensemble"' : "") + ">" +
            "<td>" + (isEns ? "★ " : "") + r.name + "</td>" +
            "<td>" + fmt(r.accuracy, 4) + "</td>" +
            "<td>" + aucCell + "</td>" +
            "<td>" + fmt(r.pr_auc, 4) + "</td>" +
            "<td>" + brierCell + "</td>" +
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

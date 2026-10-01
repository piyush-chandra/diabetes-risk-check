/* JS must equal the SHIPPED artifact, not a fresh sklearn retrain.
   train/export.py writes train/parity_fixture.json from the same fitted
   objects serialized into train/model.json. This script checks model.js
   against that fixture, plus input guards and band sanity.
   Run: node train/parity.js */
const fs = require("fs");
const vm = require("vm");

const model = JSON.parse(fs.readFileSync("train/model.json", "utf8"));
const fixture = JSON.parse(fs.readFileSync("train/parity_fixture.json", "utf8"));
const code = fs.readFileSync("site/public/model.js", "utf8");
const sandbox = { self: {}, Math, JSON, Array, Object, Number, String, Boolean, Error, Infinity, NaN, isFinite };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox);
const DM = sandbox.self.DiabetesModel;
if (!DM) { console.error("no DiabetesModel export"); process.exit(2); }
const M = DM.load(model);

let fails = 0;
function check(tag, a, b, tol) {
  const d = Math.max(...a.map((x, i) => Math.abs(x - b[i])));
  const ok = d <= tol;
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"} ${tag}: maxdiff=${d.toExponential(2)} (tol ${tol})`);
}

const jsParts = { lr: [], nb: [], knn: [], gb: [], rf: [] };
const jsStack = [];
const t0 = Date.now();
for (const row of fixture.rows) {
  const r = DM.predict(row, M);
  for (const k of Object.keys(jsParts)) jsParts[k].push(r.parts[k]);
  jsStack.push(r.prob);
}
console.log(`${fixture.rows.length} rows in ${Date.now() - t0}ms`);
for (const k of Object.keys(jsParts)) {
  check(`expert ${k}`, jsParts[k], fixture.parts[k], k === "knn" ? 2e-5 : 1e-9);
}
check("stack (shipped coefs)", jsStack, fixture.stack, 2e-5);

// artifact-shape assertions: parity is meaningless if the model changed shape
if (!model.meta || !model.meta.bands || !(model.meta.bands.low_max < model.meta.bands.high_min)) {
  fails++; console.log("FAIL meta.bands missing or crossed");
}
if (model.experts.knn.ages.length !== model.meta.n_unique) {
  fails++; console.log(`FAIL knn rows ${model.experts.knn.ages.length} != unique ${model.meta.n_unique}`);
}
if (model.experts.rf.trees.length !== 100) {
  fails++; console.log(`FAIL rf trees ${model.experts.rf.trees.length}, expected 100`);
}

let guards = 0;
const bad = [
  () => DM.predict([45, 0], M),
  () => DM.predict([NaN, 0, ...Array(14).fill(0)], M),
  () => DM.predict("nope", M),
  () => DM.predict([45, 0, ...Array(14).fill(0)], null),
];
for (const f of bad) {
  try { f(); fails++; console.log("FAIL guard: no throw"); }
  catch { guards++; }
}
console.log(`guards threw: ${guards}/4`);

// band sanity on SHIPPED weights: all-yes must outrank all-no and clear 0.5.
// all-no for a 45-year-old woman is high in this data (documented blind
// spot), so no absolute ceiling is asserted on it here.
const allNoF = DM.predict([45, 0, ...Array(14).fill(0)], M).prob;
const allNoM = DM.predict([45, 1, ...Array(14).fill(0)], M).prob;
const allYes = DM.predict([65, 1, ...Array(14).fill(1)], M).prob;
console.log(`allNoF=${allNoF.toFixed(4)} allNoM=${allNoM.toFixed(4)} allYes=${allYes.toFixed(4)}`);
if (!(allYes > allNoF && allYes > allNoM) || allYes < 0.5) { fails++; console.log("FAIL band sanity"); }

console.log(fails === 0 ? "ALL PARITY CHECKS PASSED" : `${fails} FAILURES`);
process.exit(fails === 0 ? 0 : 1);

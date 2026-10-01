/* ==========================================================================
   app.js — minimal assessment flow, zero dependencies.

   Design rules for this build:
   - Every input is a dropdown. No free text except Age.
   - The form builds itself from model.json's feature list, and a build-time
     guard makes the page refuse to run if the UI spec ever drifts from the
     model's feature order.
   - One submit → result card with plain-language explanation + next steps.
   - A tiny rule-based assistant answers follow-ups in plain words.
   - Nothing leaves the device: the model ships as a static JSON file and
     all math runs in this tab.
   ========================================================================== */
(function () {
  "use strict";

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  // ------------------------------------------------------------ feature spec
  // key = model.json feature name; label = plain-language wording shown to
  // the user; hint = the "what counts as yes" definition.
  const SYMPTOMS = [
    { key: "Polyuria", label: "Peeing much more than usual", hint: "Needing to urinate far more often, day and night" },
    { key: "Polydipsia", label: "Unquenchable thirst", hint: "Drinking far more water than normal without a clear reason" },
    { key: "sudden weight loss", label: "Sudden weight loss", hint: "Losing weight quickly without dieting" },
    { key: "weakness", label: "Unusual tiredness or weakness", hint: "Feeling drained even with basic activity" },
    { key: "Polyphagia", label: "Constantly hungry", hint: "Eating much more than usual, still feeling hungry" },
    { key: "Genital thrush", label: "Genital thrush", hint: "A fungal (yeast) infection; doctor-diagnosed" },
    { key: "visual blurring", label: "Blurred vision", hint: "Vision going in and out of focus" },
    { key: "Itching", label: "Persistent itching", hint: "Ongoing itchiness with no obvious cause" },
    { key: "Irritability", label: "Unusual irritability", hint: "Feeling on edge or short-fused" },
    { key: "delayed healing", label: "Slow-healing wounds", hint: "Cuts and grazes taking much longer than normal" },
    { key: "partial paresis", label: "Numb/weak muscles", hint: "Partial loss of muscle strength (paresis)" },
    { key: "muscle stiffness", label: "Stiff muscles", hint: "Muscles feeling tight or hard to move" },
    { key: "Alopecia", label: "Hair loss", hint: "Noticeable thinning or patchy hair loss" },
    { key: "Obesity", label: "Bulging waistline / obesity", hint: "A doctor diagnosis or body-mass above 30" },
  ];
  const AGE = "Age";
  const GENDER = "Gender";

  // Bands come from the shipped model (measured on out-of-fold scores), with
  // a hardcoded fallback if model.json predates them.
  function bands() {
    const b = MODEL && MODEL.meta && MODEL.meta.bands;
    if (b && b.low_max < b.high_min) {
      return [
        { max: b.low_max, key: "low", colour: "var(--ok)" },
        { max: b.high_min, key: "moderate", colour: "var(--warn)" },
        { max: Infinity, key: "high", colour: "var(--risk)" },
      ];
    }
    return [
      { max: 0.25, key: "low", colour: "var(--ok)" },
      { max: 0.6, key: "moderate", colour: "var(--warn)" },
      { max: Infinity, key: "high", colour: "var(--risk)" },
    ];
  }

  // ------------------------------------------------------------ state
  let MODEL = null;
  let LAST = null;

  // ------------------------------------------------------------ helpers
  function bandFor(p) {
    const list = bands();
    for (const b of list) if (p < b.max) return b;
    return list[list.length - 1];
  }

  function fmt(s) {
    return String(s);
  }

  function html(el, s) {
    el.innerHTML = String(s);
  }

  // ------------------------------------------------------------ form build
  // Renders selects for every symptom + Age/gender markup lives in HTML.
  // A dev-time guard refuses to render if the model's feature list doesn't
  // exactly match {Age, Gender} + these 14 keys.
  function buildForm() {
    if (!MODEL) return;
    const expected = new Set([AGE, GENDER]);
    SYMPTOMS.forEach((s) => expected.add(s.key));
    const mismatch = MODEL.features.filter((f) => !expected.has(f))
      .concat(SYMPTOMS.map((s) => s.key).filter((k) => !MODEL.features.includes(k)));
    if (MODEL.features.length !== expected.size || mismatch.length) {
      throw new Error("Form/model feature mismatch: " + mismatch.join(", "));
    }
    const host = $("#dynamic-groups");
    if (!host) throw new Error("missing #dynamic-groups");
    html(host, SYMPTOMS.map((s, i) => `
      <div class="field">
        <label for="sym-${i}">${s.label}</label>
        <select id="sym-${i}" data-feature="${s.key}" aria-describedby="sym-${i}-hint">
          <option value="0" selected>No</option>
          <option value="1">Yes</option>
        </select>
        <p class="hint" id="sym-${i}-hint">${s.hint}</p>
      </div>`).join(""));
  }

  // ------------------------------------------------------------ read inputs
  // ORDER IS CRITICAL. model.json's `features` array is the model's column
  // order, and readForm() must emit values in exactly that order. Each select
  // is located BY data-feature name (never by DOM position), so re-grouping
  // the display can never silently corrupt the vector.
  function readForm() {
    if (!MODEL) throw new Error("readForm called before model load");
    const model = MODEL.features;
    const raw = new Array(model.length).fill(0);
    for (let i = 0; i < model.length; i++) {
      const key = model[i];
      if (key === AGE) {
        raw[i] = getAge(); // throws if no age chosen — never silently default
      } else if (key === GENDER) {
        const g = $("#gender").value;
        if (g !== "1" && g !== "0") throw new Error("sex not chosen");
        raw[i] = g === "1" ? 1 : 0;
      } else {
        const sel = $(`select[data-feature="${key.replace(/"/g, "")}"]`);
        raw[i] = sel && sel.value === "1" ? 1 : 0;
      }
    }
    return raw;
  }

  function getAge() {
    const v = $("#age").value;
    if (v === "" || v === null) throw new Error("age not chosen");
    const age = parseInt(v, 10);
    if (!isFinite(age)) throw new Error("age not chosen");
    return Math.min(120, Math.max(1, age));
  }

  function showFormError(msg) {
    const el = $("#form-status");
    if (el) el.textContent = msg;
  }

  // ------------------------------------------------------------ predict
  function runPrediction(e) {
    e.preventDefault();
    if (!MODEL) return;
    showFormError("");
    let raw;
    try {
      raw = readForm();
    } catch (err) {
      showFormError("Please choose your age and sex before checking — the pattern can't be read without them.");
      return;
    }
    let res;
    try {
      res = window.DiabetesModel.predict(raw, MODEL);
    } catch (err) {
      console.error("predict failed", err);
      showFormError("Something went wrong calculating. Please refresh.");
      return;
    }
    const pct = Math.round(res.prob * 100);
    const band = bandFor(res.prob);
    const yesCount = raw.slice(2).filter(Boolean).length;
    const noSymptoms = yesCount === 0;
    LAST = { raw, prob: res.prob, parts: res.parts, contrib: window.DiabetesModel.contributions(raw, MODEL) };

    // headline — a pattern match against a clinic questionnaire, not a risk
    // forecast for the general population
    ($("#result-heading")).textContent =
      band.key === "low" ? "Low match" : band.key === "moderate" ? "Moderate match" : "High match";
    ($("#result-gauge")).textContent = pct + "%";
    ($("#result-gauge")).style.setProperty("color", band.colour);
    if (noSymptoms) {
      ($("#result-sub")).textContent =
        "You marked no symptoms, so this score leans almost entirely on age and sex. " +
        "In the training data, the 9 women with no marked symptoms were mostly positive " +
        "(including five identical age-35 records), while all 44 men with no symptoms were " +
        "negative — so a high score here says more about that thin slice than about you. " +
        "Treat it as unreliable either way; a blood test is the only real answer.";
    } else {
      ($("#result-sub")).textContent =
        band.key === "low"
        ? "Your symptom pattern doesn't match this clinic's diabetes cases."
        : band.key === "moderate"
        ? "Some of your answers match patterns seen in diabetes cases at this clinic. A blood test would settle it."
        : "Many of your answers match patterns commonly seen with diabetes at this clinic. Arrange a blood test.";
    }
    $("#result").hidden = false;
    $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
    renderFactors(LAST.contrib);
    renderNextSteps(band.key, noSymptoms);
    $("#ask-assistant").hidden = false;
  }

  // ------------------------------------------------------------ next steps
  function renderNextSteps(band, noSymptoms) {
    let steps;
    if (noSymptoms) {
      steps = [
        "Treat this score as unreliable — with no symptoms marked it reflects a thin, skewed slice of the training data, not you.",
        "Ask for an HbA1c blood test (no fasting needed) if you have any reason for concern — it gives a real number either way.",
        "Mention anything that changes at your next routine visit, even if this tool scored it low.",
      ];
    } else {
      steps = {
      high: [
        "Book a doctor's appointment within the next week or two — routine slot is fine, mention your symptoms.",
        "Ask for an HbA1c blood test (no fasting needed) or a fasting glucose test.",
        "Don't panic: this is a pattern match, not a diagnosis. The test gives the real answer.",
      ],
      moderate: [
        "An HbA1c blood test would settle it — quick, no fasting, gives a real number.",
        "Book a routine GP visit and mention the symptoms you marked here.",
        "If new symptoms appear before the appointment, mention those too.",
      ],
      low: [
        "No test urgently needed based on this result.",
        "If symptoms change or new ones appear, re-check here or with your doctor.",
        "Mention anything you marked at your next routine check-up.",
      ],
      }[band];
    }
    const ids = ["next-step-1", "next-step-2", "next-step-3"];
    ids.forEach((id, i) => {
      const el = document.getElementById(id);
      if (el) el.textContent = steps[i] || "";
    });
  }

  // ------------------------------------------------------------ what-if
  // Shows the 3 biggest up-drivers for the last submitted profile, including
  // sex — the dataset's base-rate shift (90% of female records positive vs
  // 45% of male) is the largest single driver and must not be hidden.
  const EXTRA_LABELS = { Age: "Age", Gender: "Sex as recorded in the data" };
  function renderFactors(contrib) {
    const list = $("#factor-list");
    if (!list) return;
    const entries = Object.keys(contrib)
      .map((k) => ({ k, d: contrib[k].delta }))
      .filter((e) => e.d > 0.002)
      .sort((a, b) => b.d - a.d)
      .slice(0, 3);
    if (!entries.length) {
      html(list, '<li class="factor-empty">No single answer dominates your score.</li>');
      return;
    }
    // build nodes and append (html() would stringify the Array)
    list.innerHTML = "";
    const maxd = entries[0].d;
    for (const e of entries) {
      const spec = SYMPTOMS.find((s) => s.key === e.k);
      const label = spec ? spec.label : (EXTRA_LABELS[e.k] || e.k);
      const pct = Math.round((e.d / maxd) * 100);
      // width set via CSSOM (CSP-safe), not an inline style attribute
      const li = document.createElement("li");
      li.innerHTML =
        '<span class="factor-name">' + label + "</span>" +
        '<span class="factor-bar"></span>';
      const fill = document.createElement("span");
      fill.className = "factor-fill";
      fill.style.width = pct + "%";
      li.querySelector(".factor-bar").appendChild(fill);
      list.appendChild(li);
    }
  }

  // ------------------------------------------------------------ assistant
  const CHIPS = [
    "What should I do next?",
    "Do I need a doctor?",
    "Which test should I ask for?",
    "Can I lower my risk?",
    "What does this score mean?",
  ];

  function askAssistant(q) {
    const log = $("#chat-log");
    const user = document.createElement("div");
    user.className = "chat-msg user-msg";
    user.textContent = q;
    log.appendChild(user);
    const bot = document.createElement("div");
    bot.className = "chat-msg bot-msg";
    bot.textContent = answerFor(q);
    log.appendChild(bot);
    log.scrollTop = log.scrollHeight;
  }

  function answerFor(q) {
    const t = String(q).toLowerCase();
    const band = LAST ? bandFor(LAST.prob).key : "low";
    const yesCount = LAST ? LAST.raw.slice(2).filter(Boolean).length : 0;
    if (/(emergency|severe|vomit|breath|confus)/.test(t)) {
      return "If you have severe vomiting, deep rapid breathing, fruity-smelling breath, or confusion, treat that as an emergency and get urgent care now. Those can be signs of diabetic ketoacidosis — a life-threatening condition this tool can't detect.";
    }
    if (/(test|hba1c|which)/.test(t) && /test/.test(t)) {
      return "Ask for an HbA1c blood test — it averages your blood sugar over roughly three months, needs no fasting, and is the standard first test. A fasting glucose test is the alternative if HbA1c isn't available.";
    }
    if (/next|step|now what/.test(t)) {
      if (band === "high") return "Book a doctor's appointment and ask for an HbA1c blood test. If that's not possible, a fasting glucose test works too. Either test will give you a clear answer rather than an estimate.";
      if (band === "moderate") return "An HbA1c blood test would settle this — it's quick, no fasting needed, and gives a real number instead of an estimate. Book it with your GP.";
      return "No test is urgently needed based on this result. If symptoms change or new ones appear, revisit the question — and mention them at your next routine check-up.";
    }
    if (/doctor|gp|appointment/.test(t)) {
      if (band === "high") return "Yes — book within the next week or two. This isn't an emergency, but it shouldn't wait months either. Ask for the soonest routine slot and mention your symptoms.";
      return "Not urgently for this result. A routine visit is enough — raise it at your next check-up, or sooner if symptoms intensify.";
    }
    if (/lower|risk|reduce|prevent/.test(t)) {
      return "The evidence-backed basics: keep your weight in a healthy range, move most days, and don't smoke. But this tool doesn't know your full picture — a doctor should guide any plan. And remember: a score is a prompt to test, not a diagnosis.";
    }
    if (/score|mean|how/.test(t)) {
      const n = (MODEL && MODEL.meta && MODEL.meta.n_unique) || "251 unique";
      return "Your score compares your answers with the patterns in " + n + " unique patient profiles from a diabetes-hospital questionnaire (520 rows with exact copies removed). It's a rough gauge, not a diagnosis — blood tests are the only real confirmation. This tool works entirely on your device; nothing was uploaded.";
    }
    if (yesCount === 0 && band === "high") {
      return "Note something important: you marked no symptoms but scored high. In the training data, all 44 men with no symptoms were negative, while 6 of the 9 women with no symptoms were positive (five of them the same age-35 record) — so this slice is thin and skewed. Trust a blood test over this result.";
    }
    return "I'm a small built-in assistant, not a doctor. I can explain the score, suggest an HbA1c blood test, or outline when to see a doctor. What would you like to know?";
  }

  // ------------------------------------------------------------ model load
  async function loadModel() {
    try {
      const res = await fetch("/model.68b2215b.json");
      if (!res.ok) throw new Error("model fetch " + res.status);
      MODEL = window.DiabetesModel.load(await res.json());
      buildForm();
      $("#model-note").textContent = "Model loaded · scored privately in your browser";
    } catch (err) {
      console.error("model load failed", err);
      const n = $("#model-note");
      n.textContent = "Couldn't load the model — check your connection.";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-small";
      btn.textContent = "Retry";
      btn.addEventListener("click", () => { btn.remove(); loadModel(); });
      n.appendChild(btn);
    }
  }

  // ------------------------------------------------------------ init
  function init() {
    loadModel();
    $("#risk-form").addEventListener("submit", runPrediction);
    $("#chat-chips"); // populated once after first submit
    const chipHost = $("#chat-chips");
    if (chipHost) {
      html(chipHost, CHIPS.map((c) => `<button type="button" class="chip">${c}</button>`).join(""));
      chipHost.addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (b) askAssistant(b.textContent);
      });
    }
    const field = $("#chat-field");
    const form = $("#chat-form");
    if (field && form) {
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const q = field.value.trim();
        if (q) { askAssistant(q); field.value = ""; }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();

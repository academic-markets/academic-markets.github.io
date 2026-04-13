/* =====================================================================
   platform.js – Interactive preference-truthfulness demo

   Probability model: logistic regression pre-fit from simulated
   competitive interview selection (300 candidates, per-department
   normalisation, sigmoid transform, top-k ranking).

   P_d = logistic(a0 + a1·fit_6q + a2·s_j + a3·fit_6q·s_j)

   The model was trained in R on the paper's actual market mechanism.
   ===================================================================== */

(function () {
  "use strict";

  // ── Constants ──────────────────────────────────────────────────────
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mobileNavBreakpoint = window.matchMedia("(min-width: 901px)");
  const PHD_RATIO_MIN = 0.5;
  const PHD_RATIO_MAX = 5.0;
  const PHD_RATIO_RANGE = PHD_RATIO_MAX - PHD_RATIO_MIN;
  const N_IDEAL = 20;          // top departments by true fit
  const QUESTION_ORDER = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"];
  const QUESTION_TITLES = {
    Q1: "Geographic Setting",
    Q2: "Region",
    Q3: "Teaching Load",
    Q4: "Course Types",
    Q5: "PhD Ratio",
    Q6: "Medical School"
  };

  // Categorical level orderings (must match R simulation)
  const LEVELS = {
    q1: ["A", "B", "C", "D"],
    q9: ["A", "B", "C", "D"],
    q10: ["A", "B", "C", "D"],
    q15: ["0", "1"]
  };

  // Map PDF Q5 categorical → numerical PhD ratio
  const Q5_MAP = { A: 1.25, B: 2.5, C: 3.75, D: null };

  // Q6 medical-school gradient scoring (matches R model training)
  // [score_at_dept_WITH_med_school, score_at_dept_WITHOUT_med_school]
  const Q6_SCORES = {
    A: [1.0,  0.0 ],   // Essential
    B: [0.85, 0.15],   // Very important
    C: [0.65, 0.35],   // Moderate
    D: [0.5,  0.5 ]    // Not important (neutral)
  };

  // Human-readable labels
  const Q_LABELS = { q1: "Geographic Setting", q2: "Region", q3: "Teaching Load", q4: "Course Types", q5: "PhD Ratio", q6: "Medical School" };
  const Q1_LABELS = { A: "Major metro", B: "Mid-sized city", C: "College town", D: "Rural", E: "No preference" };
  const Q3_LABELS = { A: "Minimal (<100 hrs)", B: "Light (100-150)", C: "Moderate (150-200)", D: "Substantial (>200)", E: "No preference" };
  const Q4_LABELS = { A: "Graduate-level", B: "Mix grad/undergrad", C: "Undergrad service", D: "Flexible" };
  const Q5_LABELS = { A: "Small (<2)", B: "Medium (2-3)", C: "Large (>3)", D: "No preference" };
  const Q6_LABELS = { A: "Essential", B: "Very important", C: "Moderate", D: "Not important" };

  // ── State ──────────────────────────────────────────────────────────
  let departments = [];
  let model = null;       // logistic model coefficients
  let trueAnswers = null;
  let reportedAnswers = null;
  let trueProb = null;
  let gaugeValueAnimationFrame = 0;
  let displayedGaugeProb = 0;

  // ── Data loading ───────────────────────────────────────────────────
  async function loadData() {
    const [deptRes, modelRes] = await Promise.all([
      fetch("data/departments.json"),
      fetch("data/interview_model.json")
    ]);
    if (!deptRes.ok || !modelRes.ok) {
      throw new Error("Unable to load demo data.");
    }
    departments = await deptRes.json();
    const modelData = await modelRes.json();
    model = modelData.model;
  }

  // ── Department-specific weights ─────────────────────────────────────
  //
  // Each department weights the 6 questions differently based on:
  //
  // Tier-based logic (via s_j):
  //   Lower-prestige departments weight geography & teaching more
  //   (location and workload are bigger parts of their value proposition).
  //   Higher-prestige departments weight PhD program fit more
  //   (research mentoring expectations are central).
  //
  // Attribute-derived adjustments:
  //   - Rural/college-town depts boost Q1 weight (location is a harder sell)
  //   - Heavy-teaching depts boost Q3 weight (need compatible candidates)
  //   - Large PhD programs boost Q5 weight (mentoring fit matters more)
  //   - Depts WITH medical schools boost Q6 weight (it's a key feature);
  //     depts WITHOUT barely weight it (irrelevant to their offer)

  const GEO_MULT = { A: 0.85, B: 1.0, C: 1.3, D: 1.5 };
  const TEACH_MULT = { A: 0.8, B: 1.0, C: 1.3, D: 1.5 };

  function deptWeights(dept) {
    const sj = dept.s_j;
    let w1 = (0.12 + 0.18 * (1 - sj)) * (GEO_MULT[dept.q1_geographic_setting] || 1);
    let w2 = 0.12 + 0.13 * (1 - sj);
    let w3 = (0.10 + 0.15 * (1 - sj)) * (TEACH_MULT[dept.q9_typical_teaching_load] || 1);
    let w4 = 0.10 + 0.08 * (1 - sj);
    const phdNorm = Math.min(Math.max((dept.q14_phd_student_ratio - 0.5) / 4.5, 0), 1);
    let w5 = (0.28 - 0.13 * (1 - sj)) * (0.7 + 0.8 * phdNorm);
    let w6 = 0.15 * (dept.q15_medical_school_proximity === 1 ? 1.4 : 0.3);
    const t = w1 + w2 + w3 + w4 + w5 + w6;
    return [w1 / t, w2 / t, w3 / t, w4 / t, w5 / t, w6 / t];
  }

  // ── Fit-score computation ──────────────────────────────────────────
  // Weighted 6-question fit score using department-specific weights.
  // Matches the R pre-computation exactly.

  function rawFitScore(answers, dept) {
    const scores = new Float64Array(6);
    const w = deptWeights(dept);

    // q1: geographic setting (ordinal, 4 levels)
    if (answers.q1 && answers.q1 !== "E") {
      scores[0] = categoricalScore(answers.q1, dept.q1_geographic_setting, LEVELS.q1);
    } else {
      scores[0] = 0.5;
    }

    // q2: region (match with signal dilution)
    if (answers.q2 && answers.q2.length > 0 && !answers.q2.includes("No preference")) {
      const k = answers.q2.length;
      scores[1] = answers.q2.includes(dept.q2_region) ? (1.0 / k) : 0.0;
    } else {
      scores[1] = 0.5;
    }

    // q3 → q9: teaching load (ordinal, 4 levels)
    if (answers.q3 && answers.q3 !== "E") {
      scores[2] = categoricalScore(answers.q3, dept.q9_typical_teaching_load, LEVELS.q9);
    } else {
      scores[2] = 0.5;
    }

    // q4 → q10: course types (ordinal A-C; D = "Flexible" is neutral)
    if (answers.q4 && answers.q4 !== "D") {
      scores[3] = categoricalScore(answers.q4, dept.q10_course_types, LEVELS.q10);
    } else {
      scores[3] = 0.5;
    }

    // q5 → q14: PhD ratio (numerical distance)
    if (answers.q5 && answers.q5 !== "D") {
      const candVal = Q5_MAP[answers.q5];
      if (candVal !== null) {
        const cn = Math.min(Math.max((candVal - PHD_RATIO_MIN) / PHD_RATIO_RANGE, 0), 1);
        const dn = Math.min(Math.max((dept.q14_phd_student_ratio - PHD_RATIO_MIN) / PHD_RATIO_RANGE, 0), 1);
        scores[4] = 1 - Math.abs(cn - dn);
      } else {
        scores[4] = 0.5;
      }
    } else {
      scores[4] = 0.5;
    }

    // q6 → q15: medical school (gradient scoring)
    if (answers.q6 && Q6_SCORES[answers.q6]) {
      const pair = Q6_SCORES[answers.q6];
      scores[5] = dept.q15_medical_school_proximity === 1 ? pair[0] : pair[1];
    } else {
      scores[5] = 0.5;
    }

    let S = 0;
    for (let i = 0; i < 6; i++) S += w[i] * scores[i];
    return S;
  }

  function categoricalScore(candVal, deptVal, levels) {
    const candPos = levels.indexOf(candVal);
    const deptPos = levels.indexOf(deptVal);
    if (candPos < 0 || deptPos < 0) return 0.5;
    if (candPos === deptPos) return 1.0;
    const maxDist = levels.length - 1;
    return Math.max(0, 1 - Math.abs(candPos - deptPos) / maxDist);
  }

  // ── Logistic model evaluation ──────────────────────────────────────
  // P(interviewed at dept) = σ(a0 + a1·fit + a2·s_j + a3·fit·s_j)

  function logistic(x) {
    return 1 / (1 + Math.exp(-x));
  }

  function pInterview(fit, sj) {
    const logit = model.intercept
               + model.coef_fit * fit
               + model.coef_sj * sj
               + model.coef_fit_sj * fit * sj;
    return logistic(logit);
  }

  // ── Aggregate interview probability ────────────────────────────────
  //
  // Three structural layers:
  //
  // 1. STRICT FILTERS on the ideal set (game-theoretic defence):
  //    - Region: department must be in candidate's true preferred regions.
  //      Prevents broadening (adding regions can't expand the ideal set).
  //    - Geographic setting: department must be within ±1 ordinal step.
  //      Prevents distant misreporting (rural ↔ major metro).
  //    - Medical school: if true pref is Essential or Very Important,
  //      department must have a medical school.
  //
  // 2. LOGISTIC MODEL: P_d = σ(a0 + a1·fit + a2·s_j + a3·fit·s_j)
  //    Trained on simulated competitive selection (300 candidates,
  //    per-department normalisation, top-k ranking).
  //
  // 3. CAP: P = min(P_reported, P_truthful)
  //    Enforces the Revelation Principle — no misreport can help.
  //    Handles residual soft-question violations.

  function computeAggregateProbability(trueAns, reportedAns) {
    const nDepts = departments.length;
    const trueFits = new Float64Array(nDepts);
    const reportedFits = new Float64Array(nDepts);

    for (let j = 0; j < nDepts; j++) {
      trueFits[j] = rawFitScore(trueAns, departments[j]);
      reportedFits[j] = rawFitScore(reportedAns, departments[j]);
    }

    // ── Build ideal set with graduated filter relaxation ──
    //
    // Filters are relaxed one at a time, from most flexible to least:
    //   Level 0: region + exact geo + medical school  (strictest)
    //   Level 1: region + ±1 geo step + medical school
    //   Level 2: region + medical school  (drop geo)
    //   Level 3: region only  (drop medical school)
    //   Level 4: top-N by fit  (drop all filters)

    const hasRegionPref = trueAns.q2 && trueAns.q2.length > 0
                       && !trueAns.q2.includes("No preference");
    const hasGeoPref    = trueAns.q1 && trueAns.q1 !== "E";
    const needsMedSchool = trueAns.q6 === "A" || trueAns.q6 === "B";
    const geoLevels = ["A", "B", "C", "D"];
    const trueGeoPos = hasGeoPref ? geoLevels.indexOf(trueAns.q1) : -1;

    function filterDepts(useRegion, geoMode, useMed) {
      const result = [];
      for (let j = 0; j < nDepts; j++) {
        const dept = departments[j];
        if (useRegion && hasRegionPref && !trueAns.q2.includes(dept.q2_region)) continue;
        if (geoMode === "exact" && hasGeoPref && dept.q1_geographic_setting !== trueAns.q1) continue;
        if (geoMode === "near" && hasGeoPref) {
          const deptGeoPos = geoLevels.indexOf(dept.q1_geographic_setting);
          if (Math.abs(trueGeoPos - deptGeoPos) > 1) continue;
        }
        if (useMed && needsMedSchool && dept.q15_medical_school_proximity !== 1) continue;
        result.push({ fit: trueFits[j], idx: j });
      }
      return result;
    }

    const relaxationLevels = [
      () => filterDepts(true, "exact", true),    // Level 0: all strict
      () => filterDepts(true, "near",  true),    // Level 1: relax geo to ±1
      () => filterDepts(true, "none",  true),    // Level 2: drop geo
      () => filterDepts(true, "none",  false),   // Level 3: drop medical school
    ];

    let idealPool = null;
    for (const tryFilter of relaxationLevels) {
      const eligible = tryFilter();
      if (eligible.length >= 5) {
        eligible.sort((a, b) => b.fit - a.fit);
        idealPool = eligible.slice(0, N_IDEAL);
        break;
      }
    }
    if (!idealPool) {
      // Level 4 fallback: top-N by fit across all departments
      const all = [];
      for (let j = 0; j < nDepts; j++) all.push({ fit: trueFits[j], idx: j });
      all.sort((a, b) => b.fit - a.fit);
      idealPool = all.slice(0, N_IDEAL);
    }

    // ── Compute weighted P(interview) at ideal departments ──
    let wSum = 0, pAggRep = 0, pAggTrue = 0;

    for (const { fit: trueFit, idx: j } of idealPool) {
      const sj = departments[j].s_j;
      const w = trueFit * trueFit;

      pAggRep  += w * pInterview(reportedFits[j], sj);
      pAggTrue += w * pInterview(trueFits[j], sj);
      wSum += w;
    }

    if (wSum === 0) return 0;

    const pReported = pAggRep / wSum;
    const pTruthful = pAggTrue / wSum;

    // Cap: truthful reporting is weakly optimal (Revelation Principle)
    return Math.min(pReported, pTruthful);
  }

  // ── UI: Questionnaire form handling ────────────────────────────────
  function readFormAnswers() {
    const form = document.getElementById("questionnaire-form");
    const q1El = form.querySelector('input[name="q1"]:checked');
    const q2Boxes = form.querySelectorAll('input[name="q2"]:checked');
    const q3El = form.querySelector('input[name="q3"]:checked');
    const q4El = form.querySelector('input[name="q4"]:checked');
    const q5El = form.querySelector('input[name="q5"]:checked');
    const q6El = form.querySelector('input[name="q6"]:checked');

    return {
      q1: q1El ? q1El.value : null,
      q2: Array.from(q2Boxes).map(cb => cb.value),
      q3: q3El ? q3El.value : null,
      q4: q4El ? q4El.value : null,
      q5: q5El ? q5El.value : null,
      q6: q6El ? q6El.value : null
    };
  }

  function writeFormAnswers(answers) {
    const form = document.getElementById("questionnaire-form");
    if (!form || !answers) return;

    form.reset();

    if (answers.q1) {
      const q1Input = form.querySelector('input[name="q1"][value="' + answers.q1 + '"]');
      if (q1Input) q1Input.checked = true;
    }

    if (Array.isArray(answers.q2)) {
      answers.q2.forEach((value) => {
        const q2Input = form.querySelector('input[name="q2"][value="' + value + '"]');
        if (q2Input) q2Input.checked = true;
      });
    }

    if (answers.q3) {
      const q3Input = form.querySelector('input[name="q3"][value="' + answers.q3 + '"]');
      if (q3Input) q3Input.checked = true;
    }

    if (answers.q4) {
      const q4Input = form.querySelector('input[name="q4"][value="' + answers.q4 + '"]');
      if (q4Input) q4Input.checked = true;
    }

    if (answers.q5) {
      const q5Input = form.querySelector('input[name="q5"][value="' + answers.q5 + '"]');
      if (q5Input) q5Input.checked = true;
    }

    if (answers.q6) {
      const q6Input = form.querySelector('input[name="q6"][value="' + answers.q6 + '"]');
      if (q6Input) q6Input.checked = true;
    }
  }

  function validateForm(answers) {
    const missing = [];
    if (!answers.q1) missing.push("Q1");
    if (answers.q2.length === 0) missing.push("Q2");
    if (!answers.q3) missing.push("Q3");
    if (!answers.q4) missing.push("Q4");
    if (!answers.q5) missing.push("Q5");
    if (!answers.q6) missing.push("Q6");
    return missing;
  }

  function setQuestionnaireAvailability(isAvailable) {
    const form = document.getElementById("questionnaire-form");
    const clearFormButton = document.getElementById("clear-form-top");
    if (!form) return;
    form.setAttribute("aria-busy", String(!isAvailable));
    form.querySelectorAll("input, button").forEach((control) => {
      control.disabled = !isAvailable;
    });
    if (clearFormButton) clearFormButton.disabled = !isAvailable || clearFormButton.hidden;
  }

  function setPlatformNavState(isOpen) {
    const navList = document.getElementById("nav-list");
    const navToggle = document.getElementById("nav-toggle");
    if (!navList || !navToggle) return;
    const isMobileNav = !mobileNavBreakpoint.matches;
    navList.classList.toggle("open", isOpen);
    navToggle.setAttribute("aria-expanded", String(isOpen));
    navList.setAttribute("aria-hidden", String(isMobileNav ? !isOpen : false));
    document.body.classList.toggle("nav-open", isMobileNav && isOpen);
  }

  // ── UI: View switching ─────────────────────────────────────────────
  function showResults(options = {}) {
    const { focusHeading: shouldFocusHeading = true } = options;
    setPlatformNavState(false);
    document.getElementById("view-questionnaire").classList.remove("platform-view--active");
    document.getElementById("view-results").classList.add("platform-view--active");
    window.scrollTo({ top: 0, behavior: prefersReducedMotion ? "auto" : "smooth" });
    if (shouldFocusHeading) focusHeading("results-title");
    else clearHeadingFocus("results-title");
  }

  function showQuestionnaire() {
    setPlatformNavState(false);
    document.getElementById("view-results").classList.remove("platform-view--active");
    document.getElementById("view-questionnaire").classList.add("platform-view--active");
    window.scrollTo({ top: 0, behavior: prefersReducedMotion ? "auto" : "smooth" });
    focusHeading("questionnaire-title");
  }

  function focusHeading(id) {
    const heading = document.getElementById(id);
    if (!heading) return;
    requestAnimationFrame(() => heading.focus({ preventScroll: true }));
  }

  function clearHeadingFocus(id) {
    const heading = document.getElementById(id);
    if (!heading) return;

    const blurHeading = () => {
      if (document.activeElement === heading) heading.blur();
    };

    requestAnimationFrame(blurHeading);
    window.setTimeout(blurHeading, 120);
  }

  function setPlatformStatus(message, type) {
    const status = document.getElementById("platform-status");
    const statusMessage = document.getElementById("platform-status-message");
    if (!status || !statusMessage) return;
    if (!message) {
      status.hidden = true;
      statusMessage.textContent = "";
      status.className = "platform-status";
      return;
    }
    status.hidden = false;
    statusMessage.textContent = message;
    status.className = "platform-status" + (type ? " platform-status--" + type : "");
  }

  function clearFormErrors() {
    const summary = document.getElementById("form-error-summary");
    if (summary) {
      summary.hidden = true;
      summary.textContent = "";
    }
    document.querySelectorAll(".q-card").forEach(card => {
      card.classList.remove("q-card--error");
      card.removeAttribute("aria-invalid");
    });
  }

  function hasAnyQuestionnaireResponses() {
    const form = document.getElementById("questionnaire-form");
    return !!form && form.querySelectorAll("input:checked").length > 0;
  }

  function updateQuestionnaireHeaderActions() {
    const clearFormButton = document.getElementById("clear-form-top");
    if (!clearFormButton) return;

    const hasResponses = hasAnyQuestionnaireResponses();
    clearFormButton.hidden = !hasResponses;
    clearFormButton.disabled = !hasResponses;
  }

  function clearQuestionnaireResponses() {
    const form = document.getElementById("questionnaire-form");
    if (!form) return;

    clearState();
    trueAnswers = null;
    reportedAnswers = null;
    trueProb = null;
    form.reset();
    clearFormErrors();
    updateQuestionnaireHeaderActions();
  }

  function showFormErrors(missing) {
    clearFormErrors();
    const cards = document.querySelectorAll(".q-card");
    missing.forEach(qn => {
      const idx = QUESTION_ORDER.indexOf(qn);
      if (cards[idx]) {
        cards[idx].classList.add("q-card--error");
        cards[idx].setAttribute("aria-invalid", "true");
      }
    });

    const summary = document.getElementById("form-error-summary");
    if (summary) {
      const labels = missing.map(qn => qn + " (" + QUESTION_TITLES[qn] + ")");
      summary.textContent = "Please complete " + labels.join(", ") + " before continuing.";
      summary.hidden = false;
      summary.focus();
    }

    const firstMissing = missing[0];
    const firstIndex = QUESTION_ORDER.indexOf(firstMissing);
    if (firstIndex >= 0 && cards[firstIndex]) {
      const firstInput = cards[firstIndex].querySelector("input");
      if (firstInput) firstInput.focus({ preventScroll: true });
    }
  }

  function setGaugeValue(prob) {
    displayedGaugeProb = prob;
    document.getElementById("gauge-value").textContent = Math.round(prob * 100) + "%";
  }

  function updateReportedProbabilityPill(prob) {
    const pill = document.getElementById("reported-live-prob");
    if (!pill) return;

    const pct = Math.round(prob * 100);
    pill.textContent = "Current probability: " + pct + "%";
    pill.className = "probability-pill";

    if (prob >= 0.5) pill.classList.add("probability-pill--high");
    else if (prob >= 0.3) pill.classList.add("probability-pill--mid");
    else pill.classList.add("probability-pill--low");
  }

  function animateGaugeValue(targetProb) {
    window.cancelAnimationFrame(gaugeValueAnimationFrame);

    if (prefersReducedMotion || Math.abs(targetProb - displayedGaugeProb) < 0.005) {
      setGaugeValue(targetProb);
      return;
    }

    const startProb = displayedGaugeProb;
    const duration = Math.max(280, Math.min(720, 340 + Math.abs(targetProb - startProb) * 1200));
    const startTime = performance.now();

    function step(now) {
      const progress = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setGaugeValue(startProb + (targetProb - startProb) * eased);

      if (progress < 1) {
        gaugeValueAnimationFrame = window.requestAnimationFrame(step);
      } else {
        gaugeValueAnimationFrame = 0;
        setGaugeValue(targetProb);
      }
    }

    gaugeValueAnimationFrame = window.requestAnimationFrame(step);
  }

  // ── Gauge rendering ────────────────────────────────────────────────
  function updateGauge(prob) {
    const gaugeFill = document.getElementById("gauge-fill");
    const totalLen = gaugeFill.getTotalLength();
    const offset = totalLen * (1 - prob);
    gaugeFill.style.strokeDasharray = totalLen;
    gaugeFill.style.strokeDashoffset = offset;

    let color;
    if (prob >= 0.5) color = "#0e7c86";
    else if (prob >= 0.3) color = "#d4a843";
    else color = "#d26a46";

    gaugeFill.style.stroke = color;
    document.getElementById("gauge-value").style.color = color;
    animateGaugeValue(prob);
    updateReportedProbabilityPill(prob);

    const comp = document.getElementById("gauge-comparison");
    if (trueProb !== null && Math.abs(prob - trueProb) > 0.002) {
      const diff = prob - trueProb;
      const absDiff = Math.abs(diff) * 100;
      const diffStr = absDiff < 1 ? "< 1" : String(Math.round(absDiff));
      if (diff < 0) {
        comp.innerHTML = '<i class="fas fa-arrow-down"></i> ' + diffStr + ' percentage point' + (absDiff >= 1.5 ? 's' : '') + ' lower than truthful reporting';
        comp.className = "gauge-comparison gauge-comparison--down";
      } else {
        comp.innerHTML = '<i class="fas fa-arrow-up"></i> ' + diffStr + ' percentage point' + (absDiff >= 1.5 ? 's' : '') + ' higher than truthful reporting';
        comp.className = "gauge-comparison gauge-comparison--up";
      }
    } else {
      comp.innerHTML = '<i class="fas fa-check"></i> Matches your truthful preferences';
      comp.className = "gauge-comparison gauge-comparison--match";
    }
  }

  // ── True preferences chips ─────────────────────────────────────────
  function renderTruePrefsChips() {
    const container = document.getElementById("true-prefs-chips");
    container.innerHTML = "";
    const items = [
      { label: Q_LABELS.q1, value: Q1_LABELS[trueAnswers.q1] || trueAnswers.q1 },
      { label: Q_LABELS.q2, value: trueAnswers.q2.includes("No preference") ? "No preference" : trueAnswers.q2.join(", ") },
      { label: Q_LABELS.q3, value: Q3_LABELS[trueAnswers.q3] || trueAnswers.q3 },
      { label: Q_LABELS.q4, value: Q4_LABELS[trueAnswers.q4] || trueAnswers.q4 },
      { label: Q_LABELS.q5, value: Q5_LABELS[trueAnswers.q5] || trueAnswers.q5 },
      { label: Q_LABELS.q6, value: Q6_LABELS[trueAnswers.q6] || trueAnswers.q6 }
    ];
    items.forEach(item => {
      const chip = document.createElement("div");
      chip.className = "pref-chip";
      chip.innerHTML = '<span class="pref-chip-label">' + item.label + '</span><span class="pref-chip-value">' + item.value + '</span>';
      container.appendChild(chip);
    });
  }

  // ── Reported preferences toggles ───────────────────────────────────
  function renderReportedGrid() {
    const grid = document.getElementById("reported-grid");
    grid.innerHTML = "";

    const questions = [
      { key: "q1", label: "Q1. Geographic Setting", type: "radio",
        options: [{ value: "A", label: "Major metro" }, { value: "B", label: "Mid-sized city" }, { value: "C", label: "College town" }, { value: "D", label: "Rural" }, { value: "E", label: "No preference" }] },
      { key: "q2", label: "Q2. Region", type: "checkbox",
        options: [{ value: "Northeast", label: "Northeast" }, { value: "Southeast", label: "Southeast" }, { value: "Midwest", label: "Midwest" }, { value: "Southwest", label: "Southwest" }, { value: "West Coast", label: "West Coast" }, { value: "No preference", label: "No preference" }] },
      { key: "q3", label: "Q3. Teaching Load", type: "radio",
        options: [{ value: "A", label: "Minimal" }, { value: "B", label: "Light" }, { value: "C", label: "Moderate" }, { value: "D", label: "Substantial" }, { value: "E", label: "No preference" }] },
      { key: "q4", label: "Q4. Course Types", type: "radio",
        options: [{ value: "A", label: "Graduate" }, { value: "B", label: "Mix" }, { value: "C", label: "Undergrad" }, { value: "D", label: "Flexible" }] },
      { key: "q5", label: "Q5. PhD Ratio", type: "radio",
        options: [{ value: "A", label: "Small (<2)" }, { value: "B", label: "Medium (2-3)" }, { value: "C", label: "Large (>3)" }, { value: "D", label: "No preference" }] },
      { key: "q6", label: "Q6. Medical School", type: "radio",
        options: [{ value: "A", label: "Essential" }, { value: "B", label: "Very important" }, { value: "C", label: "Moderate" }, { value: "D", label: "Not important" }] }
    ];

    questions.forEach(q => {
      const card = document.createElement("div");
      card.className = "toggle-card";

      const title = document.createElement("div");
      title.className = "toggle-card-title";
      title.textContent = q.label;
      card.appendChild(title);

      const optWrap = document.createElement("div");
      optWrap.className = "toggle-options";
      optWrap.setAttribute("role", "group");
      optWrap.setAttribute("aria-label", q.label);

      q.options.forEach(opt => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "toggle-btn";
        btn.textContent = opt.label;
        btn.dataset.question = q.key;
        btn.dataset.value = opt.value;

        if (q.type === "radio") {
          if (reportedAnswers[q.key] === opt.value) btn.classList.add("active");
        } else {
          if (reportedAnswers[q.key] && reportedAnswers[q.key].includes(opt.value)) btn.classList.add("active");
        }
        btn.setAttribute("aria-pressed", String(btn.classList.contains("active")));
        updateToggleDiffState(btn, q.key, opt.value, q.type);

        btn.addEventListener("click", () => {
          if (q.type === "radio") {
            reportedAnswers[q.key] = opt.value;
            optWrap.querySelectorAll(".toggle-btn").forEach(b => {
              b.classList.toggle("active", b.dataset.value === opt.value);
              b.setAttribute("aria-pressed", String(b.classList.contains("active")));
              updateToggleDiffState(b, q.key, b.dataset.value, "radio");
            });
          } else {
            const idx = reportedAnswers[q.key].indexOf(opt.value);
            if (opt.value === "No preference") {
              reportedAnswers[q.key] = idx < 0 ? ["No preference"] : [];
            } else {
              const npIdx = reportedAnswers[q.key].indexOf("No preference");
              if (npIdx >= 0) reportedAnswers[q.key].splice(npIdx, 1);
              if (idx < 0) reportedAnswers[q.key].push(opt.value);
              else reportedAnswers[q.key].splice(idx, 1);
            }
            optWrap.querySelectorAll(".toggle-btn").forEach(b => {
              b.classList.toggle("active", reportedAnswers[q.key].includes(b.dataset.value));
              b.setAttribute("aria-pressed", String(b.classList.contains("active")));
              updateToggleDiffState(b, q.key, b.dataset.value, "checkbox");
            });
          }
          recalculate();
        });

        optWrap.appendChild(btn);
      });

      card.appendChild(optWrap);
      grid.appendChild(card);
    });
  }

  function updateToggleDiffState(btn, qKey, value, type) {
    if (type === "radio") {
      const isReported = reportedAnswers[qKey] === value;
      const isTrue = trueAnswers[qKey] === value;
      btn.classList.toggle("diff", isReported && !isTrue);
    } else {
      const isReported = reportedAnswers[qKey] && reportedAnswers[qKey].includes(value);
      const isTrue = trueAnswers[qKey] && trueAnswers[qKey].includes(value);
      btn.classList.toggle("diff", isReported !== isTrue);
    }
  }

  function recalculate() {
    const prob = computeAggregateProbability(trueAnswers, reportedAnswers);
    updateGauge(prob);
    saveState();
  }

  // ── Session persistence ────────────────────────────────────────────
  function saveState() {
    sessionStorage.setItem("platform_true", JSON.stringify(trueAnswers));
    sessionStorage.setItem("platform_reported", JSON.stringify(reportedAnswers));
  }

  function clearState() {
    sessionStorage.removeItem("platform_true");
    sessionStorage.removeItem("platform_reported");
  }

  function restoreState() {
    const t = sessionStorage.getItem("platform_true");
    const r = sessionStorage.getItem("platform_reported");
    if (!t || !r) return false;
    try {
      trueAnswers = JSON.parse(t);
      reportedAnswers = JSON.parse(r);
      writeFormAnswers(trueAnswers);
      updateQuestionnaireHeaderActions();
      return true;
    } catch (error) {
      clearState();
      return false;
    }
  }

  // ── Q2 "No preference" mutual exclusion ────────────────────────────
  function setupQ2Logic() {
    const noPrefBox = document.getElementById("q2-nopref");
    const otherBoxes = document.querySelectorAll('input[name="q2"]:not(#q2-nopref)');
    noPrefBox.addEventListener("change", () => {
      if (noPrefBox.checked) otherBoxes.forEach(cb => { cb.checked = false; });
    });
    otherBoxes.forEach(cb => {
      cb.addEventListener("change", () => { if (cb.checked) noPrefBox.checked = false; });
    });
  }

  async function hydratePlatformData() {
    setQuestionnaireAvailability(false);
    setPlatformStatus("Loading demo data...", "loading");

    try {
      await loadData();
      setQuestionnaireAvailability(true);
      setPlatformStatus("");
      return true;
    } catch (error) {
      setQuestionnaireAvailability(false);
      setPlatformStatus("The demo data could not be loaded right now. Please refresh the page or try again later.", "error");
      return false;
    }
  }

  // ── Initialization ─────────────────────────────────────────────────
  async function init() {
    const navbar = document.getElementById("navbar");
    const navToggle = document.getElementById("nav-toggle");
    const navList = document.getElementById("nav-list");
    const form = document.getElementById("questionnaire-form");

    navToggle.addEventListener("click", () => {
      setPlatformNavState(!navList.classList.contains("open"));
    });

    document.addEventListener("click", (event) => {
      if (!navList.classList.contains("open")) return;
      if (navToggle.contains(event.target) || navList.contains(event.target)) return;
      setPlatformNavState(false);
    });

    const handleMobileNavBreakpointChange = (event) => {
      if (event.matches) setPlatformNavState(false);
    };

    if (typeof mobileNavBreakpoint.addEventListener === "function") {
      mobileNavBreakpoint.addEventListener("change", handleMobileNavBreakpointChange);
    } else if (typeof mobileNavBreakpoint.addListener === "function") {
      mobileNavBreakpoint.addListener(handleMobileNavBreakpointChange);
    }

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && navList.classList.contains("open")) {
        setPlatformNavState(false);
      }
    });

    window.addEventListener("scroll", () => {
      navbar.classList.toggle("sticky", window.scrollY > 24);
    }, { passive: true });
    navbar.classList.toggle("sticky", window.scrollY > 24);
    setPlatformNavState(false);

    setupQ2Logic();
    updateQuestionnaireHeaderActions();

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const answers = readFormAnswers();
      const missing = validateForm(answers);
      if (missing.length > 0) {
        showFormErrors(missing);
        return;
      }

      clearFormErrors();
      trueAnswers = JSON.parse(JSON.stringify(answers));
      reportedAnswers = JSON.parse(JSON.stringify(answers));
      enterResults();
    });
    form.addEventListener("change", () => {
      clearFormErrors();
      updateQuestionnaireHeaderActions();
    });

    const handleStartOver = () => {
      if (trueAnswers) writeFormAnswers(trueAnswers);
      updateQuestionnaireHeaderActions();
      showQuestionnaire();
    };

    document.getElementById("back-btn").addEventListener("click", handleStartOver);
    document.getElementById("back-btn-top").addEventListener("click", handleStartOver);
    document.getElementById("clear-form-top").addEventListener("click", clearQuestionnaireResponses);
    document.getElementById("reset-btn").addEventListener("click", () => {
      reportedAnswers = JSON.parse(JSON.stringify(trueAnswers));
      renderReportedGrid();
      recalculate();
    });

    const loaded = await hydratePlatformData();
    if (!loaded) return;

    if (restoreState()) {
      enterResults({ focusHeading: false });
    }
  }

  function enterResults(options = {}) {
    trueProb = computeAggregateProbability(trueAnswers, trueAnswers);
    updateGauge(computeAggregateProbability(trueAnswers, reportedAnswers));
    renderTruePrefsChips();
    renderReportedGrid();
    saveState();
    showResults(options);
  }

  document.addEventListener("DOMContentLoaded", init);
})();

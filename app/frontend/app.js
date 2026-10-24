/**
 * CarDamage AI — Dark Automotive-Tech Frontend Logic
 * Guide: CarDamage_AI_Frontend_Upgrade_Guide_v2.md
 *
 * Layout: Input 25% | Inspection Image 50% | Results 25%
 * API contracts preserved: GET /api/sample/{id}, POST /api/predict
 */
document.addEventListener("DOMContentLoaded", () => {

  // ═══ 1. STATE ═══
  const USD_TO_INR = 86.5;
  let activeCurrency = "USD";
  let selectedFile = null;
  let currentData = null;
  let originalImgSrc = "";
  let annotatedImgSrc = "";
  let activeSampleId = null;

  const PRESETS = {
    sample_1: { make:"BMW",    year:2022, body:"SEDAN",  dt:"AWD", runs:"Run & Drive Verified", title:"BMW 3-Series · 2022" },
    sample_2: { make:"TOYOTA", year:2018, body:"SUV",    dt:"FWD", runs:"Run & Drive Verified", title:"Toyota RAV4 · 2018" },
    sample_3: { make:"FORD",   year:2015, body:"PICKUP", dt:"4WD", runs:"Run & Drive Verified", title:"Ford F-150 · 2015" },
    sample_4: { make:"HONDA",  year:2020, body:"SEDAN",  dt:"FWD", runs:"DEFAULT",              title:"Honda Civic · 2020" },
  };

  const HISTORY_KEY = "cardamage_history";
  let history = loadHistory();

  // ═══ 2. DOM REFS ═══
  const views = {
    inspect:     document.getElementById("view-inspect"),
    reports:     document.getElementById("view-reports"),
    methodology: document.getElementById("view-methodology"),
    about:       document.getElementById("view-about"),
  };
  const navBtns       = document.querySelectorAll(".nav-btn");
  const mobileMenuBtn = document.getElementById("btn-mobile-menu");
  const mainNav       = document.getElementById("main-nav");

  // Stepper
  const stepperSteps  = document.querySelectorAll(".stepper-step");

  // Input
  const presetBtns    = document.querySelectorAll(".sample-btn");
  const dropzone      = document.getElementById("dropzone-area");
  const fileInput     = document.getElementById("image-file-input");
  const uploadPrompt  = document.getElementById("upload-prompt-view");
  const previewStrip  = document.getElementById("file-preview-strip");
  const previewThumb  = document.getElementById("preview-thumbnail");
  const previewName   = document.getElementById("preview-filename");
  const btnRemove     = document.getElementById("btn-remove-file");
  const form          = document.getElementById("vehicle-metadata-form");
  const makeSel       = document.getElementById("input-make");
  const yearIn        = document.getElementById("input-year");
  const bodySel       = document.getElementById("input-bodystyle");
  const dtSel         = document.getElementById("input-drivetrain");
  const opSel         = document.getElementById("input-operability");
  const btnRun        = document.getElementById("btn-run-inspection");

  // Viewer
  const emptyState    = document.getElementById("empty-state");
  const loadingOvl    = document.getElementById("loading-overlay");
  const loadingText   = document.getElementById("loading-step-text");
  const loadStages    = document.querySelectorAll(".stage");
  const imgContainer  = document.getElementById("image-container");
  const inspImg       = document.getElementById("inspection-image");
  const tabAnnotated  = document.getElementById("tab-annotated");
  const tabOriginal   = document.getElementById("tab-original");

  // Results
  const resultsPanel  = document.getElementById("results-panel");
  const resultsEmpty  = document.getElementById("results-empty");
  const resDate       = document.getElementById("res-date");
  const resVehicle    = document.getElementById("res-vehicle-title");
  const resId         = document.getElementById("res-inspection-id");
  const resCost       = document.getElementById("res-cost-val");
  const resCi         = document.getElementById("res-ci-range");
  const resSevPill    = document.getElementById("res-severity-pill");
  const resFoot       = document.getElementById("res-footprint");
  const resDsi        = document.getElementById("res-dsi-val");
  const resDmgCount   = document.getElementById("res-damage-count");
  const dmgListEl     = document.getElementById("damage-list-container");

  // Actions
  const btnReport     = document.getElementById("btn-download-report");
  const btnJson       = document.getElementById("btn-download-json");
  const btnSaveImg    = document.getElementById("btn-download-image");
  const btnNew        = document.getElementById("btn-new-analysis");

  // Currency
  const allCurrBtns   = document.querySelectorAll(".curr-btn");

  // ═══ 3. NAVIGATION ═══
  window.appSwitchView = function(v) {
    if (!views[v]) v = "inspect";
    Object.keys(views).forEach(k => views[k]?.classList.toggle("hidden", k !== v));
    Object.keys(views).forEach(k => views[k]?.classList.toggle("active", k === v));
    navBtns.forEach(b => b.classList.toggle("active", b.dataset.view === v));
    mainNav?.classList.remove("mobile-open");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  navBtns.forEach(b => b.addEventListener("click", () => appSwitchView(b.dataset.view)));
  mobileMenuBtn?.addEventListener("click", () => mainNav?.classList.toggle("mobile-open"));
  document.getElementById("btn-header-analyze")?.addEventListener("click", () => appSwitchView("inspect"));
  document.getElementById("nav-brand")?.addEventListener("click", (e) => { e.preventDefault(); appSwitchView("inspect"); });

  // ═══ 4. BENCHMARK SAMPLES ═══
  presetBtns.forEach(b => {
    b.addEventListener("click", () => {
      presetBtns.forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      activeSampleId = b.dataset.sample;
      const cfg = PRESETS[activeSampleId];
      if (cfg) {
        makeSel.value = cfg.make; yearIn.value = cfg.year;
        bodySel.value = cfg.body; dtSel.value = cfg.dt; opSel.value = cfg.runs;
      }
      clearFile();
      runSample(activeSampleId);
    });
  });

  // ═══ 5. FILE UPLOAD ═══
  dropzone?.addEventListener("click", e => {
    if (!e.target.closest("#btn-remove-file")) fileInput?.click();
  });
  dropzone?.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput?.click(); }});
  dropzone?.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone?.addEventListener("drop", e => { e.preventDefault(); dropzone.classList.remove("dragover"); if (e.dataTransfer.files?.length) handleFile(e.dataTransfer.files[0]); });
  fileInput?.addEventListener("change", e => { if (e.target.files?.length) handleFile(e.target.files[0]); });
  btnRemove?.addEventListener("click", e => { e.stopPropagation(); clearFile(); });

  function handleFile(f) {
    if (!f.type.startsWith("image/")) { alert("Unsupported format. Use JPG or PNG."); return; }
    if (f.size > 10 * 1024 * 1024) { alert("File exceeds 10 MB limit."); return; }
    selectedFile = f; activeSampleId = null;
    presetBtns.forEach(b => b.classList.remove("active"));
    const reader = new FileReader();
    reader.onload = ev => {
      originalImgSrc = ev.target.result;
      if (previewThumb) previewThumb.src = originalImgSrc;
      if (previewName) previewName.textContent = f.name;
      uploadPrompt?.classList.add("hidden");
      previewStrip?.classList.remove("hidden");
    };
    reader.readAsDataURL(f);
  }

  function clearFile() {
    selectedFile = null;
    if (fileInput) fileInput.value = "";
    previewStrip?.classList.add("hidden");
    uploadPrompt?.classList.remove("hidden");
  }

  // ═══ 6. FORM SUBMIT ═══
  form?.addEventListener("submit", e => {
    e.preventDefault();
    if (selectedFile) runUpload();
    else if (activeSampleId) runSample(activeSampleId);
    else alert("Upload an image or select a sample.");
  });

  // ═══ 7. API CALLS ═══
  async function runSample(id) {
    showLoading(true);
    setStepper("active", 2);
    try {
      const r = await fetch(`/api/sample/${id}`);
      if (!r.ok) throw new Error("Sample retrieval failed.");
      const d = await r.json();
      renderResults(d, id);
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Check backend connection.");
    } finally { showLoading(false); }
  }

  async function runUpload() {
    showLoading(true);
    setStepper("active", 2);
    const fd = new FormData();
    fd.append("file", selectedFile);
    fd.append("year", yearIn.value);
    fd.append("make", makeSel.value);
    fd.append("bodyStyle", bodySel.value);
    fd.append("drivetrain", dtSel.value);
    fd.append("runsDrives", opSel.value);
    try {
      const r = await fetch("/api/predict", { method: "POST", body: fd });
      if (!r.ok) throw new Error("Analysis request failed.");
      const d = await r.json();
      if (!d.success) throw new Error(d.error || "Analysis error.");
      renderResults(d, null);
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Try a clear vehicle image.");
    } finally { showLoading(false); }
  }

  // ═══ 8. LOADING ═══
  let loadInterval = null;
  function showLoading(on) {
    if (on) {
      emptyState?.classList.add("hidden");
      imgContainer?.classList.add("hidden");
      resultsPanel?.classList.add("hidden");
      resultsEmpty?.classList.add("hidden");
      loadingOvl?.classList.remove("hidden");
      // Make viewer area visible for overlay
      loadStages.forEach(s => s.classList.remove("active","done"));
      let cur = 0; advStage(cur);
      loadInterval = setInterval(() => {
        if (cur < loadStages.length - 1) {
          loadStages[cur]?.classList.remove("active");
          loadStages[cur]?.classList.add("done");
          cur++; advStage(cur);
        }
      }, 500);
    } else {
      clearInterval(loadInterval);
      loadStages.forEach(s => { s.classList.remove("active"); s.classList.add("done"); });
      setTimeout(() => loadingOvl?.classList.add("hidden"), 150);
    }
  }
  function advStage(i) {
    if (loadStages[i]) {
      loadStages[i].classList.add("active");
      if (loadingText) loadingText.textContent = loadStages[i].textContent.trim() + "...";
    }
  }

  // ═══ 9. STEPPER ═══
  function setStepper(state, upTo) {
    stepperSteps.forEach((s, i) => {
      s.classList.remove("active","completed");
      if (i + 1 < upTo) s.classList.add("completed");
      else if (i + 1 === upTo) s.classList.add(state);
    });
  }

  // ═══ 10. RENDER RESULTS ═══
  function renderResults(data, sampleId) {
    currentData = data;
    const rpt = data.report || {};

    // Show image
    annotatedImgSrc = data.annotated_image_base64 || "";
    if (!originalImgSrc) originalImgSrc = annotatedImgSrc;
    if (inspImg) inspImg.src = annotatedImgSrc;
    emptyState?.classList.add("hidden");
    imgContainer?.classList.remove("hidden");
    tabAnnotated?.classList.add("active");
    tabOriginal?.classList.remove("active");

    // Show results
    resultsEmpty?.classList.add("hidden");
    resultsPanel?.classList.remove("hidden");

    // Stepper completed
    setStepper("completed", 5);

    // Header
    const now = new Date();
    const dateStr = now.toLocaleDateString("en-GB", { day:"2-digit", month:"short", year:"numeric" });
    const inspId = rpt.inspection_id || `INS-${now.getFullYear()}-${String(history.length+1).padStart(4,"0")}`;
    if (resDate) resDate.textContent = dateStr;
    if (resVehicle) resVehicle.textContent = `${makeSel.value} · ${yearIn.value}`;
    if (resId) resId.textContent = inspId;

    // Cost
    renderCost(rpt);

    // Severity
    const sum = rpt.inspection_summary || {};
    let rawDsi = sum.max_dsi_score ?? 0.42;
    let dsi = rawDsi > 1 ? rawDsi / 100 : rawDsi;
    if (resDsi) resDsi.textContent = dsi.toFixed(2);

    let grade = sum.composite_severity || "moderate";
    let gl = grade.toLowerCase();
    let pill = "pill-moderate", label = "MODERATE";
    if (gl.includes("minor") || dsi < 0.28) { pill = "pill-minor"; label = "MINOR"; }
    else if (gl.includes("severe") || dsi > 0.6) { pill = "pill-severe"; label = "SEVERE"; }

    if (resSevPill) { resSevPill.textContent = label; resSevPill.className = `severity-pill ${pill}`; }
    const area = sum.total_damaged_area_pct !== undefined ? Number(sum.total_damaged_area_pct).toFixed(1) : "—";
    if (resFoot) resFoot.textContent = area !== "—" ? `${area}% surface affected` : "—";

    // Damage list
    const damages = rpt.detected_damages || rpt.damages || [];
    if (resDmgCount) resDmgCount.textContent = damages.length;
    if (dmgListEl) {
      dmgListEl.innerHTML = "";
      if (damages.length === 0) {
        dmgListEl.innerHTML = `<p class="muted" style="font-size:12px;padding:8px 0">No damage confidently detected. Try a closer image.</p>`;
      } else {
        damages.forEach((d, i) => {
          const comp = fmt(d.component || d.primary_component || "Panel");
          const typ  = fmt(d.damage_type || d.class_name || "Damage");
          const sev  = (d.severity_grade || d.severity || (dsi > .6 ? "severe" : dsi > .28 ? "moderate" : "minor")).toLowerCase();
          const conf = Math.round((d.confidence || .9) * 100);
          let sp = "pill-moderate";
          if (sev.includes("minor")) sp = "pill-minor";
          if (sev.includes("severe")) sp = "pill-severe";

          const el = document.createElement("div");
          el.className = "damage-row";
          el.innerHTML = `
            <div class="damage-row-top">
              <div><div class="damage-comp">${comp}</div><div class="damage-type">${typ}</div></div>
              <span class="severity-pill ${sp}" style="font-size:10px;padding:2px 6px">${fmt(sev)}</span>
            </div>
            <div class="damage-row-bottom">
              <span class="damage-conf">${conf}% confidence</span>
            </div>`;
          dmgListEl.appendChild(el);
        });
      }
    }

    // Save history
    const ce = rpt.repair_cost_estimate || {};
    const ptUsd = ce.estimated_cost_usd || 0;
    const ci = ce.confidence_interval_90_usd || {};
    saveHistory({
      id: inspId, date: dateStr,
      vehicle: `${makeSel.value} (${yearIn.value})`,
      areas: `${damages.length} area${damages.length===1?"":"s"}`,
      severity: label, pill,
      lowUsd: Math.round(ci.lower_bound || ptUsd * .82),
      highUsd: Math.round(ci.upper_bound || ptUsd * 1.25),
      sampleId,
    });
  }

  function fmt(s) { return s ? s.replace(/_/g," ").toLowerCase().replace(/\b\w/g,c=>c.toUpperCase()) : ""; }

  // ═══ 11. COST RENDERING ═══
  function renderCost(rpt) {
    if (!rpt) return;
    const ce = rpt.repair_cost_estimate || rpt.repair_cost_prediction || {};
    let pt = 0, lo = 0, hi = 0;
    if (ce.estimated_cost_usd !== undefined) {
      pt = Number(ce.estimated_cost_usd);
      const ci = ce.confidence_interval_90_usd || {};
      lo = ci.lower_bound !== undefined ? Number(ci.lower_bound) : pt * .82;
      hi = ci.upper_bound !== undefined ? Number(ci.upper_bound) : pt * 1.25;
    } else if (ce.estimated_cost_inr !== undefined) {
      pt = Number(ce.estimated_cost_inr) / USD_TO_INR;
      lo = pt * .82; hi = pt * 1.25;
    }
    const inr = activeCurrency === "INR";
    const m = inr ? USD_TO_INR : 1;
    const s = inr ? "₹" : "$";
    const lc = inr ? "en-IN" : "en-US";
    const L = Math.round(lo*m), H = Math.round(hi*m), P = Math.round(pt*m);
    if (resCost) resCost.textContent = `${s}${L.toLocaleString(lc)} – ${s}${H.toLocaleString(lc)}`;
    if (resCi) resCi.textContent = `Point: ${s}${P.toLocaleString(lc)} | 90% PI`;
    renderReports();
  }

  // ═══ 12. CURRENCY ═══
  function setCurrency(c) {
    activeCurrency = c;
    allCurrBtns.forEach(b => b.classList.toggle("active", b.dataset.currency === c));
    if (currentData?.report) renderCost(currentData.report);
    else renderReports();
  }
  allCurrBtns.forEach(b => b.addEventListener("click", () => setCurrency(b.dataset.currency)));

  // ═══ 13. IMAGE TOGGLE ═══
  tabAnnotated?.addEventListener("click", () => {
    tabAnnotated.classList.add("active"); tabOriginal?.classList.remove("active");
    if (annotatedImgSrc && inspImg) inspImg.src = annotatedImgSrc;
  });
  tabOriginal?.addEventListener("click", () => {
    tabOriginal.classList.add("active"); tabAnnotated?.classList.remove("active");
    if (originalImgSrc && inspImg) inspImg.src = originalImgSrc;
  });

  // ═══ 14. ACTIONS ═══
  btnReport?.addEventListener("click", () => window.print());
  btnJson?.addEventListener("click", () => {
    if (!currentData) return;
    const b = new Blob([JSON.stringify(currentData.report, null, 2)], { type: "application/json" });
    dl(b, `CarDamage_Report_${Date.now()}.json`);
  });
  btnSaveImg?.addEventListener("click", () => {
    if (!annotatedImgSrc) return;
    const a = document.createElement("a"); a.href = inspImg.src;
    a.download = `CarDamage_Annotated_${Date.now()}.jpg`; a.click();
  });
  btnNew?.addEventListener("click", () => {
    clearFile();
    emptyState?.classList.remove("hidden");
    imgContainer?.classList.add("hidden");
    resultsPanel?.classList.add("hidden");
    resultsEmpty?.classList.remove("hidden");
    setStepper("completed", 1);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  function dl(blob, name) {
    const u = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = u; a.download = name; a.click();
    URL.revokeObjectURL(u);
  }

  // ═══ 15. REPORTS HISTORY ═══
  function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; } }
  function saveHistory(rec) {
    history = [rec, ...history.filter(r => r.id !== rec.id)].slice(0, 20);
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history)); } catch {}
    renderReports();
  }

  function renderReports() {
    const tbody = document.getElementById("reports-table-body");
    const stack = document.getElementById("mobile-reports-stack");
    const empty = document.getElementById("empty-reports");
    if (!tbody || !stack) return;
    tbody.innerHTML = ""; stack.innerHTML = "";
    if (history.length === 0) { empty?.classList.remove("hidden"); return; }
    empty?.classList.add("hidden");
    const inr = activeCurrency === "INR", m = inr ? USD_TO_INR : 1, sym = inr ? "₹" : "$", lc = inr ? "en-IN" : "en-US";
    history.forEach(r => {
      const lo = Math.round((r.lowUsd||0)*m), hi = Math.round((r.highUsd||0)*m);
      const cost = `${sym}${lo.toLocaleString(lc)} – ${sym}${hi.toLocaleString(lc)}`;
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${r.date}</td><td><strong>${r.vehicle}</strong></td><td>${r.areas}</td><td><span class="severity-pill ${r.pill}" style="font-size:10px;padding:2px 6px">${r.severity}</span></td><td class="mono">${cost}</td><td>${r.sampleId?`<button class="btn btn-ghost btn-sm" onclick="appSwitchView('inspect')">View</button>`:'—'}</td>`;
      tbody.appendChild(tr);
      const card = document.createElement("div");
      card.className = "mobile-report-card";
      card.innerHTML = `<div><strong>${r.vehicle}</strong> <span class="muted" style="font-size:11px">${r.date}</span></div><div class="mt-3"><span class="severity-pill ${r.pill}" style="font-size:10px;padding:2px 6px">${r.severity}</span> ${r.areas}</div><div class="mono mt-3" style="font-weight:600">${cost}</div>`;
      stack.appendChild(card);
    });
  }

  // ═══ 16. INIT ═══
  renderReports();
  const hash = window.location.hash.replace("#","");
  if (hash && views[hash]) appSwitchView(hash);

  // Auto-load first sample
  const first = document.querySelector('.sample-btn[data-sample="sample_1"]');
  if (first) first.click();
});

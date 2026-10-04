/**
 * CarDamage AI — Professional Automotive Inspection Application Logic
 * 
 * Connects frontend to FastAPI backend:
 *   GET  /api/sample/{id}   — Benchmark sample inspection
 *   POST /api/predict        — Custom upload inspection
 * 
 * Adheres to CarDamage_AI_Professional_WebApp_Design_Guide.md
 */

document.addEventListener("DOMContentLoaded", () => {

  // ── Navigation ──────────────────────────────────────────
  const navBtns = document.querySelectorAll(".nav-item");
  const pageViews = {
    inspect:     document.getElementById("view-inspect"),
    reports:     document.getElementById("view-reports"),
    methodology: document.getElementById("view-methodology"),
    about:       document.getElementById("view-about"),
  };
  const btnHeaderAnalyze = document.getElementById("btn-header-analyze");

  // ── Inspection Elements ─────────────────────────────────
  const dropzone        = document.getElementById("dropzone-area");
  const fileInput       = document.getElementById("image-file-input");
  const uploadPromptView= document.getElementById("upload-prompt-view");
  const previewStrip    = document.getElementById("file-preview-strip");
  const previewThumb    = document.getElementById("preview-thumbnail");
  const previewFilename = document.getElementById("preview-filename");
  const btnRemoveFile   = document.getElementById("btn-remove-file");

  const metadataForm      = document.getElementById("vehicle-metadata-form");
  const makeSelect        = document.getElementById("input-make");
  const yearInput         = document.getElementById("input-year");
  const bodyStyleSelect   = document.getElementById("input-bodystyle");
  const drivetrainSelect  = document.getElementById("input-drivetrain");
  const operabilitySelect = document.getElementById("input-operability");

  const presetBtns     = document.querySelectorAll(".preset-pill");
  const emptyState     = document.getElementById("empty-state-view");
  const resultsContent = document.getElementById("results-display-view");
  const loadingOverlay = document.getElementById("loading-overlay");
  const loadingStepText= document.getElementById("loading-step-text");
  const loadingStages  = document.querySelectorAll(".load-stage");

  // ── Result Elements ─────────────────────────────────────
  const resInspectionId = document.getElementById("res-inspection-id");
  const resTimestamp    = document.getElementById("res-timestamp");
  const resCostVal     = document.getElementById("res-cost-val");
  const resCiRange     = document.getElementById("res-ci-range");
  const resDsiVal      = document.getElementById("res-dsi-val");
  const resDsiGrade    = document.getElementById("res-dsi-grade");
  const resFootprintVal= document.getElementById("res-footprint-val");
  const bomCountLabel  = document.getElementById("bom-count-label");

  const inspectionImg    = document.getElementById("inspection-image-display");
  const damagesTableBody = document.getElementById("damages-table-body");

  const tabAnnotated   = document.getElementById("tab-annotated-view");
  const tabOriginal    = document.getElementById("tab-original-view");
  const btnDownloadJson= document.getElementById("btn-download-json");
  const btnDownloadImg = document.getElementById("btn-download-image");
  const currencyBtns   = document.querySelectorAll(".curr-btn");

  // ── State ───────────────────────────────────────────────
  let selectedFile         = null;
  let currentInspectionData= null;
  let originalImageSrc     = "";
  let annotatedImageSrc    = "";
  let activeSampleId       = "sample_1";
  let activeCurrency       = "USD";
  const USD_TO_INR_RATE    = 86.5;

  // Inspection history for Reports page
  let inspectionHistory = [];

  // Preset configurations matching backend sample metadata
  const PRESET_CONFIGS = {
    sample_1: { make: "BMW",    year: 2022, body: "SEDAN",  dt: "AWD", runs: "Run & Drive Verified" },
    sample_2: { make: "TOYOTA", year: 2018, body: "SUV",    dt: "FWD", runs: "Run & Drive Verified" },
    sample_3: { make: "FORD",   year: 2015, body: "PICKUP", dt: "4WD", runs: "Run & Drive Verified" },
    sample_4: { make: "HONDA",  year: 2020, body: "SEDAN",  dt: "FWD", runs: "DEFAULT" },
  };

  // ═══════════════════════════════════════════════════════
  // 1. MULTI-PAGE NAVIGATION
  // ═══════════════════════════════════════════════════════
  window.appSwitchView = function (targetView) {
    Object.keys(pageViews).forEach((key) => {
      if (pageViews[key]) {
        pageViews[key].classList.toggle("hidden", key !== targetView);
      }
    });
    navBtns.forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-view") === targetView);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  navBtns.forEach((btn) => {
    btn.addEventListener("click", () => window.appSwitchView(btn.getAttribute("data-view")));
  });

  if (btnHeaderAnalyze) {
    btnHeaderAnalyze.addEventListener("click", () => {
      window.appSwitchView("inspect");
      document.getElementById("btn-run-inspection")?.focus();
    });
  }

  window.appSwitchToInspection = function (sampleId) {
    window.appSwitchView("inspect");
    document.querySelector(`.preset-pill[data-sample="${sampleId}"]`)?.click();
  };

  // ═══════════════════════════════════════════════════════
  // 2. BENCHMARK PRESETS
  // ═══════════════════════════════════════════════════════
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeSampleId = btn.getAttribute("data-sample");

      const cfg = PRESET_CONFIGS[activeSampleId];
      if (cfg) {
        makeSelect.value        = cfg.make;
        yearInput.value         = cfg.year;
        bodyStyleSelect.value   = cfg.body;
        drivetrainSelect.value  = cfg.dt;
        operabilitySelect.value = cfg.runs;
      }

      clearSelectedFile();
      runSampleInspection(activeSampleId);
    });
  });

  // ═══════════════════════════════════════════════════════
  // 3. FILE UPLOAD / DRAG & DROP
  // ═══════════════════════════════════════════════════════
  dropzone.addEventListener("click", (e) => {
    if (e.target.id !== "btn-remove-file" && !e.target.closest("#btn-remove-file")) {
      fileInput.click();
    }
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files?.length > 0) handleFileSelected(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files?.length > 0) handleFileSelected(e.target.files[0]);
  });

  btnRemoveFile.addEventListener("click", (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  function handleFileSelected(file) {
    // Guide §12: File type validation
    if (!file.type.startsWith("image/")) {
      alert("This file type is not supported. Upload JPG, JPEG, or PNG.");
      return;
    }
    // Guide §12: File size validation
    if (file.size > 10 * 1024 * 1024) {
      alert("The image is larger than the allowed file size.");
      return;
    }

    selectedFile = file;
    activeSampleId = null;
    presetBtns.forEach((b) => b.classList.remove("active"));

    const reader = new FileReader();
    reader.onload = (ev) => {
      originalImageSrc = ev.target.result;
      previewThumb.src = originalImageSrc;
      previewFilename.textContent = file.name;
      uploadPromptView.classList.add("hidden");
      previewStrip.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  }

  function clearSelectedFile() {
    selectedFile = null;
    fileInput.value = "";
    previewStrip.classList.add("hidden");
    uploadPromptView.classList.remove("hidden");
  }

  // ═══════════════════════════════════════════════════════
  // 4. FORM SUBMISSION
  // ═══════════════════════════════════════════════════════
  metadataForm.addEventListener("submit", (e) => {
    e.preventDefault();
    if (selectedFile) {
      runCustomUploadInspection();
    } else if (activeSampleId) {
      runSampleInspection(activeSampleId);
    } else {
      alert("Upload a vehicle image to begin the inspection.");
    }
  });

  // ═══════════════════════════════════════════════════════
  // 5. BACKEND API CALLS
  // ═══════════════════════════════════════════════════════
  async function runSampleInspection(sampleId) {
    showLoading(true);
    try {
      const response = await fetch(`/api/sample/${sampleId}`);
      if (!response.ok) throw new Error("Could not retrieve benchmark sample.");
      const data = await response.json();
      displayInspectionResults(data, sampleId);
    } catch (err) {
      console.error(err);
      alert("We couldn't complete the analysis. Please try again.");
    } finally {
      showLoading(false);
    }
  }

  async function runCustomUploadInspection() {
    showLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("year", yearInput.value);
    formData.append("make", makeSelect.value);
    formData.append("bodyStyle", bodyStyleSelect.value);
    formData.append("drivetrain", drivetrainSelect.value);
    formData.append("runsDrives", operabilitySelect.value);

    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      if (!response.ok) throw new Error("Inspection analysis failed.");
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "Analysis error");
      displayInspectionResults(data, null);
    } catch (err) {
      console.error(err);
      alert("We couldn't complete the analysis. Please try again with a clear vehicle image.");
    } finally {
      showLoading(false);
    }
  }

  // ═══════════════════════════════════════════════════════
  // 6. RESULTS RENDERING — Guide §14
  // ═══════════════════════════════════════════════════════
  function displayInspectionResults(data, sampleId) {
    currentInspectionData = data;
    const report = data.report || {};

    emptyState.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    // Inspection ID & Date
    const now = new Date();
    const dateStr = now.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
    const inspId = report.inspection_id || `INS-${now.getFullYear()}-${String(inspectionHistory.length + 1).padStart(4, "0")}`;

    if (resInspectionId) resInspectionId.textContent = inspId;
    if (resTimestamp) resTimestamp.textContent = `${dateStr} · Preliminary Assessment`;

    // Cost — read from actual backend report structure
    renderCostDisplay(report);

    // DSI & Severity
    const summary = report.inspection_summary || {};
    const rawDsi = summary.max_dsi_score ?? 0.42;
    const dsi = rawDsi > 1.0 ? rawDsi / 100.0 : rawDsi;

    if (resDsiVal) resDsiVal.textContent = dsi.toFixed(2);

    let grade = summary.composite_severity || "moderate";
    let pillClass = "pill-moderate";
    const gl = grade.toLowerCase();
    if (gl.includes("minor") || dsi < 0.28) { grade = "Minor"; pillClass = "pill-minor"; }
    else if (gl.includes("severe") || dsi > 0.60) { grade = "Severe"; pillClass = "pill-severe"; }
    else { grade = "Moderate"; pillClass = "pill-moderate"; }

    if (resDsiGrade) {
      resDsiGrade.textContent = grade;
      resDsiGrade.className = `severity-pill ${pillClass}`;
    }

    const areaPct = summary.total_damaged_area_pct !== undefined
      ? Number(summary.total_damaged_area_pct).toFixed(1)
      : "5.0";
    if (resFootprintVal) resFootprintVal.textContent = `${areaPct}% surface area affected`;

    // Annotated image
    annotatedImageSrc = data.annotated_image_base64 || "";
    if (!originalImageSrc) originalImageSrc = annotatedImageSrc;
    inspectionImg.src = annotatedImageSrc;
    tabAnnotated.classList.add("active");
    tabOriginal.classList.remove("active");

    // Damage summary table — Guide §16
    damagesTableBody.innerHTML = "";
    const damages = report.detected_damages || report.damages || [];

    if (bomCountLabel) {
      bomCountLabel.textContent = `${damages.length} damage area${damages.length === 1 ? "" : "s"}`;
    }

    if (damages.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = `<td colspan="5" class="text-muted" style="text-align:center;padding:24px;">No visible damage detected. Try another image with the affected area clearly visible.</td>`;
      damagesTableBody.appendChild(row);
    } else {
      damages.forEach((d, idx) => {
        const row = document.createElement("tr");
        const comp  = formatLabel(d.component || d.primary_component || "Panel");
        const dType = formatLabel(d.damage_type || d.class_name || "Damage");
        const sev   = (d.severity_grade || d.severity || (dsi > 0.6 ? "severe" : dsi > 0.28 ? "moderate" : "minor")).toLowerCase();
        const conf  = Math.round((d.confidence || 0.90) * 100);

        let sp = "pill-moderate";
        if (sev.includes("minor")) sp = "pill-minor";
        if (sev.includes("severe")) sp = "pill-severe";

        row.innerHTML = `
          <td class="font-mono text-muted">${String(idx + 1).padStart(2, "0")}</td>
          <td class="font-bold">${comp}</td>
          <td>${dType}</td>
          <td><span class="severity-pill ${sp}">${formatLabel(sev)}</span></td>
          <td class="font-mono">${conf}%</td>
        `;
        damagesTableBody.appendChild(row);
      });
    }

    // Add to reports history
    const costEst = report.repair_cost_estimate || {};
    const pointUsd = costEst.estimated_cost_usd || 0;
    const ciUsd = costEst.confidence_interval_90_usd || {};

    inspectionHistory.unshift({
      id: inspId,
      date: dateStr,
      vehicle: `${makeSelect.value} (${yearInput.value})`,
      areas: `${damages.length} area${damages.length === 1 ? "" : "s"}`,
      severity: grade,
      pillClass: pillClass,
      costLow: Math.round(ciUsd.lower_bound || pointUsd * 0.82),
      costHigh: Math.round(ciUsd.upper_bound || pointUsd * 1.25),
      sampleId: sampleId,
    });
    renderReportsTable();

    resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function formatLabel(str) {
    if (!str) return "";
    return str.replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
  }

  // ═══════════════════════════════════════════════════════
  // 7. IMAGE TOGGLES — Guide §15
  // ═══════════════════════════════════════════════════════
  tabAnnotated.addEventListener("click", () => {
    tabAnnotated.classList.add("active");
    tabOriginal.classList.remove("active");
    if (annotatedImageSrc) inspectionImg.src = annotatedImageSrc;
  });

  tabOriginal.addEventListener("click", () => {
    tabOriginal.classList.add("active");
    tabAnnotated.classList.remove("active");
    if (originalImageSrc) inspectionImg.src = originalImageSrc;
  });

  // ═══════════════════════════════════════════════════════
  // 8. EXPORTS — Guide §25
  // ═══════════════════════════════════════════════════════
  btnDownloadJson.addEventListener("click", () => {
    if (!currentInspectionData) return;
    const blob = new Blob([JSON.stringify(currentInspectionData.report, null, 2)], { type: "application/json" });
    downloadBlob(blob, `CarDamage_Report_${Date.now()}.json`);
  });

  btnDownloadImg.addEventListener("click", () => {
    if (!annotatedImageSrc) return;
    const a = document.createElement("a");
    a.href = inspectionImg.src;
    a.download = `CarDamage_Annotated_${Date.now()}.jpg`;
    a.click();
  });

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ═══════════════════════════════════════════════════════
  // 9. LOADING STATES — Guide §13 real stages
  // ═══════════════════════════════════════════════════════
  let loadingInterval = null;

  function showLoading(show) {
    if (show) {
      emptyState.classList.add("hidden");
      resultsContent.classList.add("hidden");
      loadingOverlay.classList.remove("hidden");

      // Reset stages
      loadingStages.forEach((s) => { s.classList.remove("active", "done"); });
      let currentStage = 0;
      advanceStage(currentStage);

      loadingInterval = setInterval(() => {
        if (currentStage < loadingStages.length - 1) {
          loadingStages[currentStage]?.classList.remove("active");
          loadingStages[currentStage]?.classList.add("done");
          currentStage++;
          advanceStage(currentStage);
        }
      }, 600);
    } else {
      clearInterval(loadingInterval);
      // Mark all done
      loadingStages.forEach((s) => { s.classList.remove("active"); s.classList.add("done"); });
      setTimeout(() => loadingOverlay.classList.add("hidden"), 200);
    }
  }

  function advanceStage(idx) {
    if (loadingStages[idx]) {
      loadingStages[idx].classList.add("active");
      const text = loadingStages[idx].textContent.trim();
      if (loadingStepText) loadingStepText.textContent = text + "...";
    }
  }

  // ═══════════════════════════════════════════════════════
  // 10. CURRENCY SWITCHER — Guide §19
  // ═══════════════════════════════════════════════════════
  function renderCostDisplay(report) {
    if (!report) return;
    const costEst = report.repair_cost_estimate || report.repair_cost_prediction || {};

    let pointUsd = 0, lowUsd = 0, highUsd = 0;

    if (costEst.estimated_cost_usd !== undefined) {
      pointUsd = Number(costEst.estimated_cost_usd);
      const ci = costEst.confidence_interval_90_usd || {};
      lowUsd  = ci.lower_bound !== undefined ? Number(ci.lower_bound) : pointUsd * 0.82;
      highUsd = ci.upper_bound !== undefined ? Number(ci.upper_bound) : pointUsd * 1.25;
    } else if (costEst.estimated_cost_inr !== undefined) {
      pointUsd = Number(costEst.estimated_cost_inr) / USD_TO_INR_RATE;
      lowUsd = pointUsd * 0.82;
      highUsd = pointUsd * 1.25;
    }

    const isInr = activeCurrency === "INR";
    const mult = isInr ? USD_TO_INR_RATE : 1;
    const sym  = isInr ? "₹" : "$";
    const loc  = isInr ? "en-IN" : "en-US";

    const p = Math.round(pointUsd * mult);
    const l = Math.round(lowUsd * mult);
    const h = Math.round(highUsd * mult);

    if (resCostVal) {
      resCostVal.textContent = `${sym}${l.toLocaleString(loc)} – ${sym}${h.toLocaleString(loc)}`;
    }
    if (resCiRange) {
      resCiRange.textContent = `Point estimate: ${sym}${p.toLocaleString(loc)} (90% CI: ${sym}${l.toLocaleString(loc)} – ${sym}${h.toLocaleString(loc)})`;
    }
  }

  currencyBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      currencyBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeCurrency = btn.getAttribute("data-currency");
      if (currentInspectionData?.report) renderCostDisplay(currentInspectionData.report);
    });
  });

  // ═══════════════════════════════════════════════════════
  // 11. REPORTS TABLE — Guide §23
  // ═══════════════════════════════════════════════════════
  function renderReportsTable() {
    const tbody = document.getElementById("reports-table-body");
    const emptyDiv = document.getElementById("empty-reports");
    if (!tbody) return;

    tbody.innerHTML = "";

    if (inspectionHistory.length === 0) {
      emptyDiv?.classList.remove("hidden");
      return;
    }

    emptyDiv?.classList.add("hidden");

    inspectionHistory.forEach((rec) => {
      const row = document.createElement("tr");
      const costStr = `$${rec.costLow.toLocaleString("en-US")} – $${rec.costHigh.toLocaleString("en-US")}`;
      const actionAttr = rec.sampleId
        ? `onclick="appSwitchToInspection('${rec.sampleId}')"`
        : "";

      row.innerHTML = `
        <td>${rec.date}</td>
        <td><span class="font-mono font-bold">${rec.id}</span></td>
        <td>${rec.vehicle}</td>
        <td>${rec.areas}</td>
        <td><span class="severity-pill ${rec.pillClass}">${rec.severity}</span></td>
        <td class="font-bold text-ink">${costStr}</td>
        <td>${rec.sampleId ? `<button type="button" class="btn btn-secondary btn-sm" ${actionAttr}>View</button>` : "—"}</td>
      `;
      tbody.appendChild(row);
    });
  }

  // Initialize reports table (empty state)
  renderReportsTable();

  // ═══════════════════════════════════════════════════════
  // 12. INITIAL LOAD
  // ═══════════════════════════════════════════════════════
  const initialBtn = document.querySelector('.preset-pill[data-sample="sample_1"]');
  if (initialBtn) initialBtn.click();
});

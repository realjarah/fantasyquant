/**
 * FantasyQuant Configure Page — league setup and session creation.
 */

(function () {
  "use strict";

  let presetData = {};
  let selectedPreset = null;

  // Uploaded file references (server-side temp paths).
  let uploadedAdpSource = null;
  let uploadedWinTotalsSource = null;
  let uploadedPropsSource = null;

  // -------------------------------------------------------------------
  // Load presets from API (to get scoring/roster details for auto-fill)
  // -------------------------------------------------------------------

  async function loadPresets() {
    try {
      const res = await fetch("/api/presets");
      presetData = await res.json();
    } catch (e) {
      console.error("Failed to load presets:", e);
    }
  }

  // -------------------------------------------------------------------
  // Preset selection
  // -------------------------------------------------------------------

  function selectPreset(name) {
    selectedPreset = name;
    document.getElementById("selected-preset").value = name;

    // Highlight selected.
    document.querySelectorAll(".preset-btn").forEach((btn) => {
      if (btn.dataset.preset === name) {
        btn.classList.add("border-fq-accent", "bg-fq-accent/10");
      } else {
        btn.classList.remove("border-fq-accent", "bg-fq-accent/10");
      }
    });

    // Auto-fill scoring/roster from preset.
    const p = presetData[name];
    if (!p) return;

    const scoring = p.scoring || {};
    const roster = p.roster || {};

    // Scoring fields.
    setSelectValue("receptions", String(scoring.receptions ?? 1.0));
    setSelectValue("passing-tds", String(scoring.passing_tds ?? 4));
    setSelectValue("te-bonus", String(scoring.te_reception_bonus ?? 0));

    // Roster fields.
    setInputValue("teams", roster.teams ?? 12);
    setInputValue("rounds", roster.rounds ?? 15);
    setInputValue("qb", roster.qb ?? 1);
    setInputValue("rb", roster.rb ?? 2);
    setInputValue("wr", roster.wr ?? 2);
    setInputValue("te", roster.te ?? 1);
    setInputValue("flex", roster.flex ?? 1);
    setInputValue("superflex", roster.superflex ?? 0);
    setInputValue("bench", roster.bench ?? 6);
    setInputValue("dst", roster.dst ?? 1);
    setInputValue("k", roster.k ?? 1);

    // Update slot max.
    updateSlotMax();
  }

  function setSelectValue(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    for (const opt of el.options) {
      if (opt.value === val) { el.value = val; return; }
    }
    // If no exact match, pick closest.
    el.value = val;
  }

  function setInputValue(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
  }

  function updateSlotMax() {
    const teams = parseInt(document.getElementById("teams").value) || 12;
    document.getElementById("slot-max").textContent = teams;
    const slotInput = document.getElementById("my-slot");
    slotInput.max = teams;
    if (parseInt(slotInput.value) > teams) slotInput.value = teams;
  }

  // -------------------------------------------------------------------
  // Data source file uploads
  // -------------------------------------------------------------------

  async function uploadFile(inputId, endpoint, statusId, refSetter) {
    const input = document.getElementById(inputId);
    const statusEl = document.getElementById(statusId);
    if (!input.files.length) return;

    const file = input.files[0];
    const formData = new FormData();
    formData.append("file", file);

    statusEl.textContent = `Uploading ${file.name}...`;
    statusEl.classList.remove("text-fq-muted");
    statusEl.classList.add("text-fq-accent");

    try {
      const res = await fetch(endpoint, { method: "POST", body: formData });
      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      refSetter(data);
      statusEl.textContent = `Uploaded: ${file.name}`;
      statusEl.classList.remove("text-fq-accent");
      statusEl.classList.add("text-fq-green");
    } catch (e) {
      statusEl.textContent = `Upload failed: ${e.message}`;
      statusEl.classList.remove("text-fq-accent");
      statusEl.classList.add("text-fq-red");
    }
  }

  // -------------------------------------------------------------------
  // Form submission
  // -------------------------------------------------------------------

  async function createSession() {
    const statusEl = document.getElementById("launch-status");
    const btn = document.getElementById("launch-btn");
    btn.disabled = true;
    statusEl.textContent = "Creating session...";
    statusEl.classList.remove("hidden");

    const body = {
      preset: selectedPreset || null,
      my_slot: parseInt(document.getElementById("my-slot").value) || 1,
      scoring: {
        receptions: parseFloat(document.getElementById("receptions").value),
        passing_tds: parseFloat(document.getElementById("passing-tds").value),
        te_reception_bonus: parseFloat(document.getElementById("te-bonus").value),
        interceptions: -2.0,
        rushing_yards: 0.1,
        rushing_tds: 6.0,
        receiving_yards: 0.1,
        receiving_tds: 6.0,
        fumbles_lost: -2.0,
      },
      roster: {
        teams: parseInt(document.getElementById("teams").value),
        rounds: parseInt(document.getElementById("rounds").value),
        qb: parseInt(document.getElementById("qb").value),
        rb: parseInt(document.getElementById("rb").value),
        wr: parseInt(document.getElementById("wr").value),
        te: parseInt(document.getElementById("te").value),
        flex: parseInt(document.getElementById("flex").value),
        superflex: parseInt(document.getElementById("superflex").value),
        bench: parseInt(document.getElementById("bench").value),
        dst: parseInt(document.getElementById("dst").value),
        k: parseInt(document.getElementById("k").value),
      },
    };

    // Data sources.
    const oddsKey = document.getElementById("odds-api-key").value.trim();
    if (oddsKey) body.odds_api_key = oddsKey;
    if (uploadedAdpSource) body.adp_source = uploadedAdpSource;
    if (uploadedWinTotalsSource) body.win_totals_source = uploadedWinTotalsSource;
    if (uploadedPropsSource) body.player_props_source = uploadedPropsSource;

    try {
      const res = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to create session");
      }

      statusEl.textContent = "Loading projections — redirecting to draft room...";

      // In dev mode, go directly to the draft room.
      // In production, this would go through Stripe first.
      const payRes = await fetch(`/api/sessions/${data.session_id}/pay`, { method: "POST" });
      const payData = await payRes.json();
      window.location.href = payData.checkout_url;
    } catch (e) {
      statusEl.textContent = `Error: ${e.message}`;
      btn.disabled = false;
    }
  }

  // -------------------------------------------------------------------
  // Event listeners
  // -------------------------------------------------------------------

  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => selectPreset(btn.dataset.preset));
  });

  document.getElementById("teams").addEventListener("change", updateSlotMax);

  document.getElementById("config-form").addEventListener("submit", (e) => {
    e.preventDefault();
    createSession();
  });

  // Data sources toggle.
  document.getElementById("data-toggle").addEventListener("click", () => {
    const panel = document.getElementById("data-sources-panel");
    const btn = document.getElementById("data-toggle");
    panel.classList.toggle("hidden");
    btn.innerHTML = panel.classList.contains("hidden")
      ? "Advanced: use your own data &#9654;"
      : "Advanced: use your own data &#9660;";
  });

  // File upload handlers.
  document.getElementById("adp-file").addEventListener("change", () => {
    uploadFile("adp-file", "/api/upload/adp", "adp-status", (data) => {
      uploadedAdpSource = data.adp_source;
    });
  });

  document.getElementById("wintotals-file").addEventListener("change", () => {
    uploadFile("wintotals-file", "/api/upload/win-totals", "wintotals-status", (data) => {
      uploadedWinTotalsSource = data.win_totals_source;
    });
  });

  document.getElementById("props-file").addEventListener("change", () => {
    uploadFile("props-file", "/api/upload/props", "props-status", (data) => {
      uploadedPropsSource = data.player_props_source;
    });
  });

  // -------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------

  loadPresets();
  updateSlotMax();
})();

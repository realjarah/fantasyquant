/**
 * FantasyQuant Draft Room — client-side logic.
 *
 * Manages the draft state, player pool, pick input, solver
 * recommendations, and roster display via the REST API.
 */

(function () {
  "use strict";

  const app = document.getElementById("draft-app");
  const SESSION_ID = app.dataset.sessionId;
  const MY_SLOT = parseInt(app.dataset.mySlot);
  const TOTAL_TEAMS = parseInt(app.dataset.totalTeams);
  const TOTAL_ROUNDS = parseInt(app.dataset.totalRounds);
  const API = `/api/sessions/${SESSION_ID}`;

  // Position colors.
  const POS_COLORS = {
    QB: "bg-red-900/30 text-red-400",
    RB: "bg-green-900/30 text-green-400",
    WR: "bg-blue-900/30 text-blue-400",
    TE: "bg-amber-900/30 text-amber-400",
  };

  // State.
  let currentState = null;
  let currentFilter = "";
  let currentSearch = "";
  let pollTimer = null;

  // -------------------------------------------------------------------
  // API helpers
  // -------------------------------------------------------------------

  async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(`${API}${path}`, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `API error ${res.status}`);
    }
    return res.json();
  }

  // -------------------------------------------------------------------
  // State polling
  // -------------------------------------------------------------------

  async function refreshState() {
    try {
      currentState = await api("GET", "/state");
      updateUI();
    } catch (e) {
      console.error("State refresh failed:", e);
    }
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(refreshState, 2000);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  // -------------------------------------------------------------------
  // UI updates
  // -------------------------------------------------------------------

  function updateUI() {
    if (!currentState) return;

    const { status, current_pick, current_round, total_rounds, is_my_turn, picks, my_roster, roster_needs } = currentState;

    // Loading state.
    const loadingBadge = document.getElementById("loading-badge");
    const clockBadge = document.getElementById("clock-badge");

    if (status === "loading") {
      loadingBadge.classList.remove("hidden");
      clockBadge.classList.add("hidden");
      return;
    }
    loadingBadge.classList.add("hidden");

    // Pick/round counters.
    document.getElementById("pick-counter").textContent = `#${current_pick}`;
    document.getElementById("round-counter").textContent = `${current_round} / ${total_rounds}`;

    // Clock indicator.
    if (is_my_turn && status !== "complete") {
      clockBadge.classList.remove("hidden");
      document.getElementById("auto-pick-btn").classList.remove("hidden");
      fetchRecommendation();
    } else {
      clockBadge.classList.add("hidden");
      document.getElementById("auto-pick-btn").classList.add("hidden");
      document.getElementById("rec-banner").classList.add("hidden");
    }

    // Draft log.
    renderDraftLog(picks);

    // Roster.
    renderRoster(my_roster);

    // Roster needs.
    renderRosterNeeds(roster_needs);

    // Player pool.
    refreshPlayers();

    // Complete state.
    if (status === "complete") {
      stopPolling();
      document.getElementById("view-summary-btn").classList.remove("hidden");
    }
  }

  function renderDraftLog(picks) {
    const log = document.getElementById("draft-log");
    if (!picks.length) return;

    let html = "";
    let lastRound = 0;
    for (const p of picks) {
      if (p.round !== lastRound) {
        html += `<div class="text-xs text-fq-muted font-semibold uppercase tracking-wider mt-3 mb-1">Round ${p.round}</div>`;
        lastRound = p.round;
      }
      const posClass = POS_COLORS[p.position] || "bg-fq-card text-fq-muted";
      const mine = p.is_mine;
      html += `
        <div class="flex items-center gap-3 py-1 px-2 rounded text-sm ${mine ? "bg-fq-accent/10 border border-fq-accent/20" : "hover:bg-fq-card/50"}">
          <span class="text-xs font-mono text-fq-muted w-8">${p.overall_pick}.</span>
          <span class="text-xs font-mono px-1.5 py-0.5 rounded ${posClass}">${p.position}</span>
          <span class="${mine ? "font-semibold text-slate-100" : "text-slate-300"}">${p.player_name}</span>
          <span class="text-xs text-fq-muted ml-auto">${p.team}</span>
          ${mine ? '<span class="text-xs text-fq-accent">YOU</span>' : ""}
        </div>`;
    }
    log.innerHTML = html;
    log.scrollTop = log.scrollHeight;
  }

  function renderRoster(roster) {
    const panel = document.getElementById("roster-panel");
    if (!roster.length) {
      panel.innerHTML = '<div class="text-sm text-fq-muted text-center py-4">No picks yet</div>';
      return;
    }

    // Group by position.
    const groups = { QB: [], RB: [], WR: [], TE: [] };
    for (const p of roster) {
      if (groups[p.position]) groups[p.position].push(p);
    }

    let html = "";
    for (const [pos, players] of Object.entries(groups)) {
      if (!players.length) continue;
      const posClass = POS_COLORS[pos] || "";
      html += `<div>
        <div class="text-xs font-semibold uppercase tracking-wider text-fq-muted mb-1">${pos}</div>`;
      for (const p of players) {
        html += `
          <div class="flex items-center justify-between py-1 text-sm">
            <div class="flex items-center gap-2">
              <span class="text-xs font-mono px-1.5 py-0.5 rounded ${posClass}">${pos}</span>
              <span class="truncate">${p.player_name}</span>
            </div>
            <span class="text-xs text-fq-muted font-mono">${Math.round(p.total_projected)}</span>
          </div>`;
      }
      html += `</div>`;
    }
    panel.innerHTML = html;
  }

  function renderRosterNeeds(needs) {
    const el = document.getElementById("roster-needs");
    const entries = Object.entries(needs);
    if (!entries.length) {
      el.innerHTML = '<div class="text-fq-green">Starter slots filled</div>';
      return;
    }
    el.innerHTML = entries
      .map(([pos, n]) => `<div>Need: <span class="font-semibold text-fq-amber">${n}x ${pos}</span></div>`)
      .join("");
  }

  // -------------------------------------------------------------------
  // Player pool
  // -------------------------------------------------------------------

  async function refreshPlayers() {
    const params = new URLSearchParams();
    if (currentFilter) params.set("position", currentFilter);
    if (currentSearch) params.set("search", currentSearch);
    params.set("limit", "80");

    try {
      const data = await api("GET", `/players?${params}`);
      renderPlayerList(data.players, data.total_available);
    } catch (e) {
      console.error("Player refresh failed:", e);
    }
  }

  function renderPlayerList(players, total) {
    const list = document.getElementById("player-list");
    if (!players.length) {
      list.innerHTML = '<div class="text-sm text-fq-muted text-center py-4">No players found</div>';
      return;
    }

    let html = `<div class="text-xs text-fq-muted px-3 py-1">${total} available</div>`;
    for (const p of players) {
      const posClass = POS_COLORS[p.position] || "";
      html += `
        <div class="player-row flex items-center gap-2 px-3 py-1.5 hover:bg-fq-card/50 cursor-pointer text-sm"
             data-pid="${p.player_id}" data-name="${p.player_name}">
          <span class="text-xs font-mono px-1.5 py-0.5 rounded ${posClass}">${p.position}</span>
          <span class="truncate flex-1">${p.player_name}</span>
          <span class="text-xs text-fq-muted">${p.team}</span>
          <span class="text-xs font-mono text-fq-muted w-12 text-right">${Math.round(p.total_projected)}</span>
        </div>`;
    }
    list.innerHTML = html;

    // Click to draft.
    for (const row of list.querySelectorAll(".player-row")) {
      row.addEventListener("click", () => {
        const pid = row.dataset.pid;
        const name = row.dataset.name;
        document.getElementById("pick-input").value = name;
      });
    }
  }

  // -------------------------------------------------------------------
  // Recommendation
  // -------------------------------------------------------------------

  async function fetchRecommendation() {
    try {
      const rec = await api("GET", "/recommend");
      showRecommendation(rec);
    } catch (e) {
      // Not our turn or still loading — hide banner.
      document.getElementById("rec-banner").classList.add("hidden");
    }
  }

  function showRecommendation(rec) {
    const banner = document.getElementById("rec-banner");
    banner.classList.remove("hidden");
    document.getElementById("rec-name").textContent = rec.player_name;
    document.getElementById("rec-pos").textContent = rec.position;
    document.getElementById("rec-team").textContent = rec.team;
    document.getElementById("rec-wins").textContent = rec.expected_wins;
    document.getElementById("rec-pts").textContent = Math.round(rec.expected_points);

    // Alternatives.
    const altsEl = document.getElementById("rec-alts");
    const altsNames = document.getElementById("rec-alt-names");
    if (rec.alternatives && rec.alternatives.length) {
      altsEl.classList.remove("hidden");
      altsNames.textContent = rec.alternatives.map((a) => `${a.player_name} (${a.position})`).join(", ");
    } else {
      altsEl.classList.add("hidden");
    }

    // Accept button picks the recommended player.
    document.getElementById("rec-accept").onclick = () => makePick(rec.player_id);
  }

  // -------------------------------------------------------------------
  // Pick submission
  // -------------------------------------------------------------------

  async function makePick(playerIdOrName) {
    const errorEl = document.getElementById("pick-error");
    errorEl.classList.add("hidden");

    const body = {};
    // If it looks like a player_id (alphanumeric with dashes), use that.
    if (/^[0-9\-]+$/.test(playerIdOrName)) {
      body.player_id = playerIdOrName;
    } else {
      body.player_name = playerIdOrName;
    }

    try {
      const res = await api("POST", "/pick", body);
      if (!res.success) {
        errorEl.textContent = res.error || "Pick failed";
        errorEl.classList.remove("hidden");
        return;
      }
      document.getElementById("pick-input").value = "";
      await refreshState();
    } catch (e) {
      errorEl.textContent = e.message;
      errorEl.classList.remove("hidden");
    }
  }

  async function autoPick() {
    const errorEl = document.getElementById("pick-error");
    errorEl.classList.add("hidden");
    try {
      const res = await api("POST", "/auto-pick");
      if (!res.success) {
        errorEl.textContent = res.error || "Auto pick failed";
        errorEl.classList.remove("hidden");
        return;
      }
      document.getElementById("pick-input").value = "";
      await refreshState();
    } catch (e) {
      errorEl.textContent = e.message;
      errorEl.classList.remove("hidden");
    }
  }

  // -------------------------------------------------------------------
  // Event listeners
  // -------------------------------------------------------------------

  document.getElementById("pick-btn").addEventListener("click", () => {
    const val = document.getElementById("pick-input").value.trim();
    if (val) makePick(val);
  });

  document.getElementById("pick-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const val = e.target.value.trim();
      if (val) makePick(val);
    }
  });

  document.getElementById("auto-pick-btn").addEventListener("click", autoPick);

  // Position filters.
  for (const btn of document.querySelectorAll(".pos-filter")) {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".pos-filter").forEach((b) => b.classList.remove("active", "border-fq-accent", "text-fq-accent"));
      btn.classList.add("active", "border-fq-accent", "text-fq-accent");
      currentFilter = btn.dataset.pos;
      refreshPlayers();
    });
  }

  // Search.
  let searchTimer;
  document.getElementById("player-search").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      currentSearch = e.target.value.trim();
      refreshPlayers();
    }, 300);
  });

  // Summary button.
  document.getElementById("view-summary-btn").addEventListener("click", () => {
    window.location.href = `/summary/${SESSION_ID}`;
  });

  // -------------------------------------------------------------------
  // Init
  // -------------------------------------------------------------------

  refreshState();
  startPolling();
})();

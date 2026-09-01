/* Fault detail panel, shared by the motor history page and the analyse results.

   The reference content itself is injected per page from faults.py -- this file only
   renders it. Keeping the prose in Python and the rendering here means the text exists in
   exactly one place, the same reason LOCATION_LABELS stopped being restated in JavaScript.

   Markup comes from templates/_fault_panel.html, which both pages include. */
(function () {
  let FAULTS = {};
  let LABELS = {};
  let RECURRENCE = {};
  let TOTAL_RECORDINGS = 0;
  let MOTOR_NAME = null;

  let panel, backdrop, titleEl, bodyEl, closeBtn;
  let lastTrigger = null;

  const esc = (s) => String(s).replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const para = (text) => `<p>${esc(text)}</p>`;
  const section = (heading, html) => `<h3>${esc(heading)}</h3>${html}`;

  function init(config) {
    FAULTS = config.faults || {};
    LABELS = config.labels || {};
    RECURRENCE = config.recurrence || {};
    TOTAL_RECORDINGS = config.totalRecordings || 0;
    MOTOR_NAME = config.motorName || null;

    panel = document.getElementById("fault-panel");
    backdrop = document.getElementById("fault-backdrop");
    if (!panel) return;               // page did not include the markup
    titleEl = document.getElementById("fault-panel-title");
    bodyEl = document.getElementById("fault-panel-body");
    closeBtn = document.getElementById("fault-panel-close");

    closeBtn.addEventListener("click", close);
    backdrop.addEventListener("click", close);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !panel.hidden) close();
    });
  }

  /* context: { confidence, recorded } -- both optional. `trigger` is the element that
     opened the panel, so focus can be handed back to it on close. */
  function open(location, context, trigger) {
    const f = FAULTS[location];
    if (!f || !panel) return;
    context = context || {};

    const seen = RECURRENCE[location];

    titleEl.textContent = LABELS[location] || location;

    // This finding first, then how it behaves across the motor's history, then the
    // reference text. Specific before general -- the reader already knows which fault
    // they clicked, so leading with the definition wastes the first line.
    const parts = [para(f.summary)];

    const context_lines = [];
    if (context.confidence != null) {
      context_lines.push(
        `Flagged at ${esc(context.confidence)}% confidence` +
        (context.recorded ? ` on the ${esc(context.recorded)} recording.` : " on this recording.")
      );
    }
    if (seen && TOTAL_RECORDINGS > 1) {
      context_lines.push(
        `Seen in ${seen.count} of this motor's ${TOTAL_RECORDINGS} recordings, ` +
        `first on ${esc(seen.first)}` +
        (seen.count > 1 ? " — persistent, not a one-off reading." : ".")
      );
    }
    if (context_lines.length) {
      parts.push(section(TOTAL_RECORDINGS > 1 ? "On this motor" : "This finding",
        context_lines.map((c) => `<p>${c}</p>`).join("")));
    }

    const r = f.reliability;
    parts.push(section("How reliable is this detection",
      `<p class="fault-metrics">precision ${r.precision.toFixed(2)} · ` +
      `recall ${r.recall.toFixed(2)} · F1 ${r.f1.toFixed(2)}</p>` + para(r.note)));

    parts.push(section("What the system measured", para(f.measured)));
    parts.push(section("What to do now", para(f.action)));

    // `causes`, `if_ignored` and `prevention` are deliberately not rendered: they are
    // generic reference that made the panel long enough to stop being read, where
    // everything above is specific to this machine and this reading.

    bodyEl.innerHTML = parts.join("");
    bodyEl.scrollTop = 0;             // reopening on a different fault should start at the top
    panel.hidden = false;
    backdrop.hidden = false;
    lastTrigger = trigger || null;
    closeBtn.focus();
  }

  function close() {
    if (!panel) return;
    panel.hidden = true;
    backdrop.hidden = true;
    // Focus goes back where it came from, or keyboard users are dumped at the page top.
    if (lastTrigger) { lastTrigger.focus(); lastTrigger = null; }
  }

  window.FaultPanel = { init, open, has: (location) => !!FAULTS[location] };
})();

/* FleetSense -- transparent 3D replica of the 2.2 kW three-phase asynchronous motor the
   dataset was recorded on. Blueprint aesthetic: near-invisible fills with petrol edge
   outlines, so the internals (rotor, windings, bearings, air gap) read through the
   housing. Parts light up by severity once an analysis returns.

   Every one of the nine detectable locations maps onto a real physical part:
     bearing_outer / bearing_inner / bearing_ball -> the three elements of a rolling bearing
     rotor_bar                                    -> rotor cage bars
     winding                                      -> stator winding
     bend                                         -> shaft
     static_/dynamic_eccentricity                 -> air gap (a condition of the gap
                                                     between rotor and stator, not a part
                                                     you can replace -- shown as the
                                                     annular volume itself)
     voltage_unbalance                            -> terminal box (supply side; not a
                                                     mechanical fault at all)

   Bearings: the analysis does not identify WHICH end (drive vs non-drive) a bearing fault
   sits on, so both bearings are flagged together and the tooltip says so rather than
   implying we know. */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const PETROL = 0x009999;

const SEVERITY_COLOR = {
  low: 0xe6b800,      // yellow  -- low severity
  unknown: 0xe07a1f,  // amber   -- detected, severity not assessable
  high: 0xdc3c2e,     // red     -- high severity
};

const SEVERITY_LABEL = {
  low: "Low severity",
  unknown: "Detected — severity not assessable",
  high: "High severity",
};

/* part id -> which analysis locations flag it */
const PART_LOCATIONS = {
  bearing_outer: ["bearing_outer"],
  bearing_ball: ["bearing_ball"],
  bearing_inner: ["bearing_inner"],
  rotor_bars: ["rotor_bar"],
  winding: ["winding"],
  shaft: ["bend"],
  air_gap: ["static_eccentricity", "dynamic_eccentricity"],
  terminal_box: ["voltage_unbalance"],
};

const PART_LABEL = {
  housing: "Stator housing",
  cooling_fins: "Cooling fins",
  terminal_box: "Terminal box",
  endbell_de: "Drive-end shield",
  fan_cowl: "Fan cowl",
  feet: "Mounting feet",
  eyebolt: "Lifting eye",
  winding: "Stator winding",
  rotor: "Rotor core",
  rotor_bars: "Rotor bars",
  air_gap: "Air gap",
  shaft: "Shaft",
  bearing_outer: "Bearing — outer race",
  bearing_ball: "Bearing — rolling elements",
  bearing_inner: "Bearing — inner race",
};

const BEARING_PARTS = new Set(["bearing_outer", "bearing_ball", "bearing_inner"]);

/* Picking priority. A plain "nearest hit wins" raycast is useless here: the housing
   encloses everything, so it would swallow every hover and no internal component could
   ever be inspected -- which is the entire point of a see-through replica. Instead the
   most specific part under the cursor wins, so the shell only gets picked where nothing
   is behind it. Higher number = wins. */
const PICK_PRIORITY = {
  housing: 0, cooling_fins: 0, endbell_de: 0, fan_cowl: 0, feet: 0, eyebolt: 0,
  winding: 1, terminal_box: 1,
  air_gap: 2, rotor: 2,
  shaft: 3,
  rotor_bars: 4,
  bearing_outer: 5, bearing_inner: 5, bearing_ball: 5,
};

let renderer, scene, camera, controls, raycaster, container, tooltipEl, detailEl;
let parts = {};          // id -> { group, fills[], edges[], status, issue }
let pickables = [];      // meshes eligible for raycasting
let hoveredId = null, pinnedId = null;
const pointer = new THREE.Vector2();
let pointerInside = false;

/* ---------- geometry helpers ---------- */

function makePart(id) {
  const group = new THREE.Group();
  scene.add(group);
  parts[id] = { group, fills: [], edges: [], status: null, issue: null };
  return parts[id];
}

/** Adds a solid: a barely-there fill (for raycasting + volume) plus petrol edge lines. */
function addSolid(id, geometry, { position, rotation, fill = true, threshold = 15 } = {}) {
  const part = parts[id] || makePart(id);

  if (fill) {
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color: PETROL,
        transparent: true,
        opacity: 0.07,
        depthWrite: false,        // transparent shells shouldn't occlude each other
        side: THREE.DoubleSide,
      })
    );
    mesh.userData.partId = id;
    if (position) mesh.position.set(...position);
    if (rotation) mesh.rotation.set(...rotation);
    part.group.add(mesh);
    part.fills.push(mesh);
    pickables.push(mesh);
  }

  const lines = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry, threshold),
    new THREE.LineBasicMaterial({ color: PETROL, transparent: true, opacity: 0.85 })
  );
  if (position) lines.position.set(...position);
  if (rotation) lines.rotation.set(...rotation);
  part.group.add(lines);
  part.edges.push(lines);

  return part;
}

/* Cylinders are Y-aligned in three.js; the motor axis is X, hence the z-rotation. */
const AXIAL = [0, 0, Math.PI / 2];

function buildMotor() {
  // --- housing ---
  addSolid("housing", new THREE.CylinderGeometry(1.0, 1.0, 2.4, 48), { rotation: AXIAL });

  // cooling fins: edge-only, they'd read as noise if filled
  const finCount = 20;
  for (let i = 0; i < finCount; i++) {
    const a = (i / finCount) * Math.PI * 2;
    // skip the arc under the terminal box
    if (Math.abs(a - Math.PI / 2) < 0.34) continue;
    const fin = new THREE.BoxGeometry(2.3, 0.13, 0.05);
    addSolid("cooling_fins", fin, {
      position: [0, Math.sin(a) * 1.05, Math.cos(a) * 1.05],
      rotation: [-a, 0, 0],
      fill: false,
    });
  }

  // --- terminal box (supply side -> voltage unbalance) ---
  addSolid("terminal_box", new THREE.BoxGeometry(0.9, 0.5, 0.72), { position: [0.05, 1.22, 0] });

  // --- end shields / cowl ---
  addSolid("endbell_de", new THREE.CylinderGeometry(0.78, 0.88, 0.32, 40), {
    position: [1.36, 0, 0], rotation: AXIAL,
  });
  addSolid("fan_cowl", new THREE.CylinderGeometry(0.92, 0.86, 0.5, 40), {
    position: [-1.45, 0, 0], rotation: AXIAL,
  });

  // --- feet ---
  addSolid("feet", new THREE.BoxGeometry(1.95, 0.12, 0.3), { position: [0, -1.06, 0.62] });
  addSolid("feet", new THREE.BoxGeometry(1.95, 0.12, 0.3), { position: [0, -1.06, -0.62] });

  // --- lifting eye ---
  addSolid("eyebolt", new THREE.TorusGeometry(0.12, 0.028, 8, 20), { position: [-0.62, 1.13, 0] });

  // --- stator winding: laminated core + the two end-winding bundles ---
  addSolid("winding", new THREE.CylinderGeometry(0.84, 0.84, 1.72, 40), { rotation: AXIAL });
  addSolid("winding", new THREE.TorusGeometry(0.74, 0.1, 10, 32), {
    position: [0.9, 0, 0], rotation: [0, Math.PI / 2, 0],
  });
  addSolid("winding", new THREE.TorusGeometry(0.74, 0.1, 10, 32), {
    position: [-0.9, 0, 0], rotation: [0, Math.PI / 2, 0],
  });

  // --- air gap: the annulus between rotor OD and stator bore ---
  addSolid("air_gap", new THREE.CylinderGeometry(0.63, 0.63, 1.78, 40), { rotation: AXIAL });

  // --- rotor core ---
  addSolid("rotor", new THREE.CylinderGeometry(0.57, 0.57, 1.8, 40), { rotation: AXIAL });

  // --- rotor cage bars ---
  const barCount = 16;
  for (let i = 0; i < barCount; i++) {
    const a = (i / barCount) * Math.PI * 2;
    addSolid("rotor_bars", new THREE.CylinderGeometry(0.035, 0.035, 1.86, 8), {
      position: [0, Math.sin(a) * 0.5, Math.cos(a) * 0.5],
      rotation: AXIAL,
      threshold: 30,
    });
  }

  // --- shaft (bent-shaft fault) ---
  addSolid("shaft", new THREE.CylinderGeometry(0.16, 0.16, 3.9, 24), {
    position: [0.35, 0, 0], rotation: AXIAL,
  });

  // --- bearings at both ends; which end is NOT identified by the analysis ---
  for (const x of [1.12, -1.12]) {
    addSolid("bearing_outer", new THREE.TorusGeometry(0.29, 0.04, 10, 28), {
      position: [x, 0, 0], rotation: [0, Math.PI / 2, 0],
    });
    addSolid("bearing_inner", new THREE.TorusGeometry(0.2, 0.035, 10, 28), {
      position: [x, 0, 0], rotation: [0, Math.PI / 2, 0],
    });
    const ballCount = 9;
    for (let i = 0; i < ballCount; i++) {
      const a = (i / ballCount) * Math.PI * 2;
      addSolid("bearing_ball", new THREE.SphereGeometry(0.045, 12, 10), {
        position: [x, Math.sin(a) * 0.245, Math.cos(a) * 0.245],
        threshold: 40,
      });
    }
  }
}

/* ---------- appearance ---------- */

function paint(id) {
  const part = parts[id];
  if (!part) return;

  const flagged = part.status !== null;
  const isActive = id === hoveredId || id === pinnedId;
  const color = flagged ? SEVERITY_COLOR[part.status] : PETROL;

  const edgeOpacity = flagged ? 1.0 : isActive ? 1.0 : 0.85;
  const fillOpacity = flagged ? (isActive ? 0.3 : 0.22) : isActive ? 0.18 : 0.07;

  for (const e of part.edges) {
    e.material.color.setHex(color);
    e.material.opacity = edgeOpacity;
  }
  for (const f of part.fills) {
    f.material.color.setHex(color);
    f.material.opacity = fillOpacity;
  }
}

function repaintAll() {
  Object.keys(parts).forEach(paint);
}

/* ---------- interaction ---------- */

function describe(id) {
  const part = parts[id];
  const label = PART_LABEL[id] || id;
  if (!part || part.status === null) {
    return { label, status: "No fault detected", tone: "ok", extra: null };
  }
  const pct = Math.round((part.issue.presence_confidence || 0) * 100);
  return {
    label,
    status: SEVERITY_LABEL[part.status],
    tone: part.status,
    extra: `${pct}% confidence` + (BEARING_PARTS.has(id)
      ? " · the analysis does not identify which end (drive or non-drive)" : ""),
  };
}

function updateTooltip(clientX, clientY) {
  if (!hoveredId) { tooltipEl.hidden = true; return; }
  const d = describe(hoveredId);
  tooltipEl.hidden = false;
  tooltipEl.className = "motor-tip tone-" + d.tone;
  tooltipEl.innerHTML =
    '<div class="motor-tip-name"></div><div class="motor-tip-status"></div>';
  tooltipEl.querySelector(".motor-tip-name").textContent = d.label;
  tooltipEl.querySelector(".motor-tip-status").textContent = d.status;

  const r = container.getBoundingClientRect();
  const x = Math.min(clientX - r.left + 14, r.width - 210);
  const y = Math.min(clientY - r.top + 14, r.height - 60);
  tooltipEl.style.left = Math.max(6, x) + "px";
  tooltipEl.style.top = Math.max(6, y) + "px";
}

function updateDetail() {
  const id = pinnedId || hoveredId;
  if (!id) {
    detailEl.className = "motor-detail";
    detailEl.textContent = "Hover a component to inspect it. Click to pin. Drag or use the arrow keys to rotate, scroll to zoom.";
    return;
  }
  const d = describe(id);
  detailEl.className = "motor-detail tone-" + d.tone;
  detailEl.textContent = d.label + " — " + d.status + (d.extra ? " · " + d.extra : "");
}

function pick() {
  if (!pointerInside) return null;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(pickables, false);
  if (!hits.length) return null;

  // Most specific part wins, not the nearest -- see PICK_PRIORITY. Distance only
  // breaks ties within the same priority band.
  let best = null, bestRank = -1;
  for (const hit of hits) {
    const id = hit.object.userData.partId;
    const rank = PICK_PRIORITY[id] ?? 0;
    if (rank > bestRank) { bestRank = rank; best = id; }
  }
  return best;
}

/* ---------- keyboard orbit ---------- */

/* OrbitControls' own key handling is no use here: its arrow keys call pan(), not rotate,
   and enablePan is off anyway -- so the built-in listenToKeyEvents() would do nothing.
   There are getAzimuthalAngle()/getPolarAngle() getters but no setters, so the camera is
   moved directly around controls.target in spherical coordinates instead. */
const _offset = new THREE.Vector3();
const _spherical = new THREE.Spherical();
const KEY_STEP = THREE.MathUtils.degToRad(5);

function orbitBy(dTheta, dPhi) {
  _offset.copy(camera.position).sub(controls.target);
  _spherical.setFromVector3(_offset);
  _spherical.theta += dTheta;
  // Respect the controls' own polar limits, and never reach the poles exactly -- at
  // phi 0 or PI the view flips over.
  _spherical.phi = THREE.MathUtils.clamp(
    _spherical.phi + dPhi,
    Math.max(controls.minPolarAngle, 0.001),
    Math.min(controls.maxPolarAngle, Math.PI - 0.001)
  );
  _offset.setFromSpherical(_spherical);
  camera.position.copy(controls.target).add(_offset);
  camera.lookAt(controls.target);
  controls.update();   // damping is on, so the controls must resync with the moved camera
}

function onKeyDown(e) {
  let dTheta = 0, dPhi = 0;
  if (e.key === "ArrowLeft") dTheta = -KEY_STEP;
  else if (e.key === "ArrowRight") dTheta = KEY_STEP;
  else if (e.key === "ArrowUp") dPhi = -KEY_STEP;
  else if (e.key === "ArrowDown") dPhi = KEY_STEP;
  else return;
  e.preventDefault();   // otherwise the page scrolls while rotating
  orbitBy(dTheta, dPhi);
}

/* ---------- public API ---------- */

function init(el, tip, detail) {
  container = el; tooltipEl = tip; detailEl = detail;

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(3.4, 2.1, 4.6);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 2.6;
  controls.maxDistance = 14;
  controls.enablePan = false;

  raycaster = new THREE.Raycaster();

  buildMotor();
  repaintAll();
  updateDetail();

  new ResizeObserver(resize).observe(container);
  resize();

  // Listens on the container (which carries tabindex="0") rather than window, so the
  // arrow keys only steer the motor when the viewer actually has focus.
  container.addEventListener("keydown", onKeyDown);

  renderer.domElement.addEventListener("pointermove", (e) => {
    const r = container.getBoundingClientRect();
    pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    pointerInside = true;
    const next = pick();
    if (next !== hoveredId) {
      const prev = hoveredId;
      hoveredId = next;
      if (prev) paint(prev);
      if (next) paint(next);
      updateDetail();
    }
    updateTooltip(e.clientX, e.clientY);
  });

  renderer.domElement.addEventListener("pointerleave", () => {
    pointerInside = false;
    const prev = hoveredId;
    hoveredId = null;
    if (prev) paint(prev);
    tooltipEl.hidden = true;
    updateDetail();
  });

  renderer.domElement.addEventListener("click", () => {
    const prev = pinnedId;
    pinnedId = hoveredId && hoveredId === pinnedId ? null : hoveredId;
    if (prev) paint(prev);
    if (pinnedId) paint(pinnedId);
    updateDetail();
  });

  loop();
}

function resize() {
  if (!container) return;
  const w = container.clientWidth || 1;
  const h = container.clientHeight || 1;
  // updateStyle left on (default) deliberately: it sets an explicit CSS size on the
  // canvas. Without it the element lays out at its *buffer* size, which setPixelRatio
  // multiplies by the device ratio -- on a 2x display the canvas would render at twice
  // the container width and burst the panel.
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
}

/** issues: the array straight off /analyze */
function update(issues) {
  for (const id of Object.keys(parts)) {
    parts[id].status = null;
    parts[id].issue = null;
  }

  for (const issue of issues || []) {
    for (const [partId, locations] of Object.entries(PART_LOCATIONS)) {
      if (!locations.includes(issue.location) || !parts[partId]) continue;
      const status = issue.severity === "high" ? "high"
        : issue.severity === "low" ? "low"
          : "unknown";
      // a part covering two locations (air gap) keeps the more serious of them
      const rank = { low: 1, unknown: 2, high: 3 };
      if (parts[partId].status === null || rank[status] > rank[parts[partId].status]) {
        parts[partId].status = status;
        parts[partId].issue = issue;
      }
    }
  }

  pinnedId = null;
  repaintAll();
  updateDetail();
}

function resetView() {
  camera.position.set(3.4, 2.1, 4.6);
  controls.target.set(0, 0, 0);
  controls.update();
}

/** Debug aid: the actual painted state of every part, straight off the materials. */
function inspect() {
  const out = { _camera: {
    x: +camera.position.x.toFixed(3),
    y: +camera.position.y.toFixed(3),
    z: +camera.position.z.toFixed(3),
    azimuthDeg: +THREE.MathUtils.radToDeg(controls.getAzimuthalAngle()).toFixed(2),
    polarDeg: +THREE.MathUtils.radToDeg(controls.getPolarAngle()).toFixed(2),
    distance: +controls.getDistance().toFixed(3),
  } };
  for (const [id, p] of Object.entries(parts)) {
    out[id] = {
      status: p.status,
      edgeColor: "#" + p.edges[0].material.color.getHexString(),
      fillOpacity: p.fills.length ? +p.fills[0].material.opacity.toFixed(3) : null,
    };
  }
  return out;
}

window.MotorViewer = { init, update, resetView, inspect };

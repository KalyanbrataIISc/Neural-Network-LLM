import { Network } from "./network.js";
import { CaptureGame, WIDTH, HEIGHT, DIRECTIONS, CTF_SPEC, CTF_ENV, CTF_REWARDS, CTF_INPUT_NAMES, CTF_OUTPUT_NAMES, captureInputs, captureTransition } from "./ctf-game.js";
import { NetworkMap } from "./network-map.js";
import { MAP_SPLITS, MAP_PROTOCOL, mapSplit, sampleMapId } from "./ctf-maps.js";
import { RewardEditor } from "./reward-editor.js";

const $ = (id) => document.getElementById(id);
const ctx = $("game").getContext("2d");
const requestedMap = Number(new URL(location.href).searchParams.get("map") || MAP_SPLITS.test.start);
const initialMap = Number.isInteger(requestedMap) && requestedMap >= 1 && requestedMap <= MAP_PROTOCOL.total ? requestedMap : MAP_SPLITS.test.start;
const game = new CaptureGame(Math.random, initialMap);
const rewardEditor = new RewardEditor($("reward-fields"), CTF_REWARDS, "robot-capture-rewards-v1");
const STORAGE = "robot-capture-model-v1";
let network = new Network(null, Math.random, CTF_ENV.defaultShape, CTF_SPEC);
let model = { ...network.serialize(), untrained: true }, worker = null, mode = "human", last = 0, elapsed = 0, uiTicks = 0;
let liveRewards = rewardEditor.read(), noticeTimer;
const map = new NetworkMap($("network-map"), {
  getNetwork: () => network,
  inputNames: CTF_INPUT_NAMES, outputNames: CTF_OUTPUT_NAMES, maxOverviewEdges: 2500,
  commit: (next) => {
    if (worker) { stopTraining(); $("train-status").textContent = "Training stopped for a graph edit"; }
    install({ ...next.serialize(), method: "manual-experiment" });
  },
  notify: (message) => notice(message),
});
$("open-training").onclick = () => $("training-settings").showModal();
$("close-training").onclick = () => $("training-settings").close();
$("train-shortcut").onclick = () => $("train").click();
$("watch-shortcut").onclick = () => $("watch").click();
$("edit-brain").onclick = () => $("inspect").click();
$("graph-step").onclick = () => { $("graph-speed").value = "0"; map.step(captureInputs(game)); };
const keys = new Set();
const touchPointers = new Map();
for (const button of document.querySelectorAll("[data-keys]")) {
  button.onpointerdown = (event) => {
    if (mode !== "human") return;
    event.preventDefault();
    button.setPointerCapture(event.pointerId);
    touchPointers.set(event.pointerId, button.dataset.keys.split(" ").map((key) => key === "fire" ? " " : key));
    if (game.state === "ready" || game.state === "paused") start();
  };
  const release = (event) => touchPointers.delete(event.pointerId);
  button.onpointerup = release;
  button.onpointercancel = release;
  button.onlostpointercapture = release;
}
function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => $("notice").hidden = true, 7000);
}
function install(data, persist = true) {
  const loaded = Network.load(data, CTF_SPEC);
  network = loaded;
  model = data;
  map.reset();
  if (persist) {
    try { localStorage.setItem(STORAGE, JSON.stringify(data)); }
    catch { notice("Model is usable, but browser storage is unavailable. Export it to keep a copy."); }
  }
  $("experiment-shape").textContent = network.shape.join(" → ");
  $("hidden-count").textContent = network.shape.slice(1, -1).join(" / ");
  const count = network.weights.flat(2).length;
  $("graph-structure").textContent = `${network.shape.join(" → ")} · ${count.toLocaleString()} parameters`;
  $("parameter-count").textContent = `${count.toLocaleString()} parameters`;
  $("model-info").textContent = data.method === "double-dqn" && Number.isFinite(data.episodes) && Number.isFinite(data.score)
    ? `Saved training run: ${data.episodes.toLocaleString()} episodes. Saved evaluation: ${data.score.toFixed(1)} mission points. Continued training starts from these weights.`
    : data.untrained ? "Random weights, no pretrained policy. Train the robot to learn this mission; navigating and fighting may need substantial experimentation." : "Unmeasured policy. Weights are manually edited; train and evaluate to measure their performance.";
  $("map-training-info").textContent = data.mapProtocol?.version === MAP_PROTOCOL.version
    ? `${new Set(data.trainingMapIds || []).size} distinct training maps recorded in this model's run history. Validation maps: 1,001–1,005. Held-out test maps: 2,001–10,000.`
    : "Map-training history is not recorded for these weights. New runs use only maps 1–1,000; validation uses 1,001–1,005. Imported weights may have an unknown history.";
  $("checkpoint-rewards").textContent = data.untrained ? "Random weights; no training run yet." : rewardEditor.describe(data.rewardConfig);
}
try {
  const saved = localStorage.getItem(STORAGE);
  install(saved ? JSON.parse(saved) : model, false);
} catch { install(model, false); notice("Saved robot weights could not be loaded. Using a new random network."); }
function setMode(next) {
  mode = next;
  keys.clear();
  touchPointers.clear();
  for (const id of ["human", "ai"]) {
    $(id).classList.toggle("selected", id === mode);
    $(id).setAttribute("aria-pressed", String(id === mode));
  }
  $("live-label").innerHTML = `<i></i> ${mode === "human" ? "HUMAN" : "NEURAL NETWORK"} IN CONTROL`;
}
$("human").onclick = () => setMode("human");
$("ai").onclick = () => setMode("ai");
function start() {
  if (["won", "lost", "timeout"].includes(game.state)) game.reset();
  game.state = "playing";
  keys.clear();
  $("game").focus();
  sync();
}
$("play").onclick = start;
$("watch").onclick = () => { if ($("training-settings").open) $("training-settings").close(); setMode("ai"); game.reset(); start(); };
$("restart").onclick = () => { game.reset(); keys.clear(); sync(); };
$("pause").onclick = () => {
  if (game.state === "playing") { game.state = "paused"; keys.clear(); sync(); }
  else start();
};
const controlKeys = ["w", "a", "s", "d", "arrowup", "arrowleft", "arrowdown", "arrowright", " ", "shift"];
window.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.target.closest("input,textarea,button,select,a,dialog,.neural-panel")) return;
  const key = event.key.toLowerCase();
  if (controlKeys.includes(key)) {
    event.preventDefault();
    keys.add(key);
  }
});
window.addEventListener("keyup", (event) => keys.delete(event.key.toLowerCase()));
function pauseOnBlur() {
  keys.clear();
  touchPointers.clear();
  if (game.state === "playing") { game.state = "paused"; sync(); }
}
window.addEventListener("blur", pauseOnBlur);
document.addEventListener("visibilitychange", () => { if (document.hidden) pauseOnBlur(); });
function humanAction() {
  const held = (key) => keys.has(key) || [...touchPointers.values()].some((pressed) => pressed.includes(key));
  const dx = Number(held("d") || held("arrowright")) - Number(held("a") || held("arrowleft"));
  const dy = Number(held("s") || held("arrowdown")) - Number(held("w") || held("arrowup"));
  const ability = held("shift") ? 2 : held(" ") ? 1 : 0;
  if (!dx && !dy && !ability) return null;
  const vector = dx || dy ? [dx / Math.hypot(dx, dy), dy / Math.hypot(dx, dy)] : game.player.facing;
  const direction = DIRECTIONS.findIndex(([x, y]) => Math.abs(x - vector[0]) < 0.01 && Math.abs(y - vector[1]) < 0.01);
  return ability * 8 + Math.max(0, direction);
}
function showMap() {
  const range = MAP_SPLITS[mapSplit(game.mapId)];
  $("map-set").value = game.map.split;
  $("map-id").min = range.start; $("map-id").max = range.end; $("map-id").value = game.mapId;
  $("map-info").textContent = `Map ${game.mapId.toLocaleString()} / 10,000 · ${range.label} · seed ${game.map.seed} · changing maps restarts the mission`;
  $("game").dataset.mapId = game.mapId;
}
function loadMap(id) {
  try {
    const split = mapSplit(id);
    if (split !== $("map-set").value) throw new Error("Choose a map ID inside the selected map set.");
    const wasPlaying = game.state === "playing";
    game.reset(id); if (wasPlaying) game.state = "playing";
    keys.clear(); touchPointers.clear(); map.reset();
    const url = new URL(location.href); url.searchParams.set("map", id); history.replaceState(null, "", url);
    showMap(); sync();
  } catch (error) { notice(error.message); }
}
$("map-controls").onsubmit = (event) => { event.preventDefault(); loadMap(Number($("map-id").value)); };
$("map-set").onchange = () => loadMap(MAP_SPLITS[$("map-set").value].start);
$("map-random").onclick = () => {
  const split = $("map-set").value, range = MAP_SPLITS[split];
  let id = sampleMapId(Math.random, split);
  if (id === game.mapId) id = id === range.end ? range.start : id + 1;
  loadMap(id);
};
function sync() {
  const p = game.player;
  $("score").textContent = game.score;
  $("health").textContent = p.health;
  $("resources").textContent = `${p.ammo} / ${Math.floor(p.energy)}`;
  $("objective").textContent = game.carrying ? "RETURN HOME" : "GET FLAG";
  $("overlay").hidden = game.state === "playing";
  $("pause").textContent = game.state === "playing" ? "Ⅱ Pause" : "▷ Resume";
  const text = {
    ready: ["ONE ROBOT. TWO OBJECTIVES.", "Out. And back.", "Take the orange flag. Return to your cyan base alive.", "Start mission ↗"],
    paused: ["MISSION PAUSED", "Take a moment.", "Your robot is waiting. Resume when you are ready.", "Resume mission ↗"],
    won: ["FLAG DELIVERED", "Home, with the flag.", "Mission complete. Try a different policy or reward configuration.", "New mission ↗"],
    lost: ["ROBOT DESTROYED", "A different route?", "Use cover, collect supplies, and choose when to engage.", "Try again ↗"],
    timeout: ["TIME LIMIT REACHED", "The clock ran out.", "Missions last 3,600 physics frames. Try a faster route home.", "Try again ↗"],
  }[game.state];
  if (text) ["overlay-tag", "overlay-title", "overlay-copy", "play"].forEach((id, i) => $(id).textContent = text[i]);
}
function disc(x, y, radius, fill) {
  ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fillStyle = fill; ctx.fill();
}
function draw() {
  const { base: BASE, flag: FLAG, walls: WALLS } = game;
  ctx.clearRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = "#a5b89a12";
  for (let x = 16; x < WIDTH; x += 24) for (let y = 16; y < HEIGHT; y += 24) ctx.fillRect(x, y, 1, 1);
  ctx.strokeStyle = "#73d9d6"; ctx.lineWidth = 2;
  ctx.strokeRect(BASE.x - 28, BASE.y - 28, 56, 56);
  ctx.fillStyle = "#73d9d620"; ctx.fillRect(BASE.x - 28, BASE.y - 28, 56, 56);
  ctx.font = "12px monospace"; ctx.fillStyle = "#73d9d6"; ctx.fillText("HOME", BASE.x - 16, BASE.y + 46);
  for (const wall of WALLS) {
    ctx.fillStyle = "#2b3930"; ctx.fillRect(wall.x, wall.y, wall.w, wall.h);
    ctx.strokeStyle = "#4e6551"; ctx.strokeRect(wall.x, wall.y, wall.w, wall.h);
  }
  if (!game.carrying) {
    ctx.strokeStyle = "#ffba75"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(FLAG.x, FLAG.y + 18); ctx.lineTo(FLAG.x, FLAG.y - 18); ctx.stroke();
    ctx.fillStyle = "#ffba75"; ctx.beginPath(); ctx.moveTo(FLAG.x, FLAG.y - 18); ctx.lineTo(FLAG.x + 22, FLAG.y - 10); ctx.lineTo(FLAG.x, FLAG.y - 1); ctx.fill();
    ctx.fillText("FLAG", FLAG.x - 16, FLAG.y + 36);
  }
  const colors = { health: "#b9f577", ammo: "#f1d17d", energy: "#73d9d6" };
  for (const supply of game.supplies) {
    ctx.globalAlpha = supply.cooldown ? 0.2 : 1;
    ctx.fillStyle = colors[supply.type]; ctx.fillRect(supply.x - 9, supply.y - 9, 18, 18);
    ctx.fillStyle = "#142018"; ctx.textAlign = "center"; ctx.fillText({ health: "+", ammo: "A", energy: "E" }[supply.type], supply.x, supply.y + 4); ctx.textAlign = "start";
  }
  ctx.globalAlpha = 1;
  for (const enemy of game.enemies) if (enemy.health > 0) {
    disc(enemy.x, enemy.y, 12, "#ed877c"); disc(enemy.x, enemy.y, 4, "#432526");
    ctx.fillStyle = "#7b4540"; ctx.fillRect(enemy.x - 14, enemy.y - 20, 28, 3);
    ctx.fillStyle = "#ffb1a8"; ctx.fillRect(enemy.x - 14, enemy.y - 20, 28 * enemy.health / 60, 3);
  }
  for (const bullet of game.bullets) disc(bullet.x, bullet.y, 3, bullet.enemy ? "#ff998d" : "#dcff9f");
  const p = game.player;
  disc(p.x, p.y, 14, game.carrying ? "#ffba75" : "#b9f577"); disc(p.x, p.y, 7, "#23352b");
  ctx.strokeStyle = "#e5f6cf"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x + p.facing[0] * 23, p.y + p.facing[1] * 23); ctx.stroke();
  if (game.carrying) { ctx.fillStyle = "#ffba75"; ctx.fillText("⚑", p.x - 4, p.y - 24); }
  ctx.fillStyle = "#8ca080"; ctx.font = "11px monospace"; ctx.fillText(`${Math.max(0, 60 - game.frames / 60).toFixed(1)}s remaining`, 18, 24);
}
function frame(time) {
  elapsed += Math.min((time - last) / 1000, 0.1); last = time;
  while (elapsed >= 1 / 15) {
    elapsed -= 1 / 15;
    if (game.state === "playing") {
      try { liveRewards = rewardEditor.read(); } catch { /* Keep the last valid preview config. */ }
      const action = mode === "ai" ? network.action(captureInputs(game)) : humanAction();
      const result = captureTransition(game, action, liveRewards);
      if (game.state === "playing" && game.frames >= 3600) game.state = "timeout";
      $("live-breakdown").textContent = CTF_REWARDS.map(({ key, label }) => `${label}: ${result.terms[key].toFixed(4)}`).join(" · ");
      $("live-reward").textContent = `Live reward preview: ${result.reward.toFixed(4)} this action · current editor coefficients; this does not train the model.`;
      $("action-readout").textContent = `${mode === "ai" ? "AI" : "Human"} action: ${action === null ? "idle" : CTF_OUTPUT_NAMES[action]} · ${game.hits} hits · ${game.kills} defeats`;
      sync();
    }
  }
  draw();
  if (uiTicks % 6 === 0 && !worker) { network.predict(captureInputs(game)); map.capture(network); }
  map.draw(time, Number($("graph-speed").value));
  if (++uiTicks % 15 === 0) {
    $("train-shortcut").textContent = worker ? "□ Stop training" : "✧ Train robot";
    $("network-training-status").textContent = `${$("train-status").textContent} · ${$("epoch").textContent} episodes · validation points: ${$("evaluation").textContent}`;
    const values = captureInputs(game);
    if ($("sensor-details").open) $("sensor-values").textContent = CTF_INPUT_NAMES.map((name, i) => `${String(i + 1).padStart(2)}  ${name.padEnd(30)} ${values[i].toFixed(4)}`).join("\n");
    if ($("output-values").parentElement.open) {
      const q = network.predict(values);
      $("output-values").textContent = CTF_OUTPUT_NAMES.map((name, i) => `${String(i + 1).padStart(2)}  ${name.padEnd(22)} ${q[i].toFixed(5)}`).join("\n");
    }
  }
  requestAnimationFrame(frame);
}
function stopTraining() {
  worker?.terminate(); worker = null;
  $("train").textContent = "✧ Train robot ↗";
  $("epochs").disabled = false; $("fresh").disabled = false; $("policy-selection").disabled = false;
  rewardEditor.lock(false);
  map.endTraining();
}
$("train").onclick = () => {
  if (worker) { stopTraining(); $("train-status").textContent = "Stopped; latest saved checkpoint retained"; return; }
  try {
    const rewardConfig = rewardEditor.save();
    const budget = Number($("epochs").value);
    worker = new Worker(new URL("./train-worker.js", import.meta.url), { type: "module" });
    $("epochs").disabled = true; $("fresh").disabled = true; $("policy-selection").disabled = true;
    rewardEditor.lock(true);
    $("train").textContent = "□ Stop training";
    $("train-status").textContent = "Starting simulations…";
    $("reward-average").textContent = "Mean episode reward: —";
    $("epoch").textContent = `0 / ${budget.toLocaleString()}`;
    $("progress").style.width = "0%";
    worker.onmessage = ({ data }) => {
      if (data.type === "progress") {
        if (data.snapshot) map.training(data.snapshot);
        if (data.trainingMapCount !== undefined) $("map-training-info").textContent = `${data.trainingMapCount} distinct training maps visited this run/history. Training pool: 1–1,000; validation: 1,001–1,005; test maps 2,001–10,000 are excluded.`;
        $("train-status").textContent = "Simulating at full speed";
        $("epoch").textContent = `${data.episode.toLocaleString()} / ${data.episodes.toLocaleString()}`;
        $("progress").style.width = `${data.episode / data.episodes * 100}%`;
        $("throughput").textContent = `${data.framesPerSecond.toLocaleString()} frames / sec`;
        $("exploration").textContent = `Exploration: ${Math.round(data.epsilon * 100)}%`;
        $("evaluation").textContent = data.score.toFixed(1);
        $("reward-breakdown").textContent = CTF_REWARDS.map(({ key, label }) => `${label}: ${data.rewardTerms[key].toFixed(3)}`).join(" · ");
        $("reward-average").textContent = data.episode ? `Mean episode reward: ${data.reward.toFixed(3)} (last ${Math.min(data.episode, 100)} episodes).` : "Mean episode reward: waiting for completed episodes…";
      }
      if (data.type === "checkpoint" || data.type === "complete") {
        try { install(data.model); }
        catch (error) { notice(error.message); stopTraining(); $("train-status").textContent = "Could not load training checkpoint"; return; }
      }
      if (data.type === "complete") { stopTraining(); $("train-status").textContent = "Complete · selected checkpoint saved"; notice("Training complete. Watch AI play to evaluate the policy yourself."); }
      if (data.type === "error") { stopTraining(); $("train-status").textContent = "Training failed"; notice(data.message); }
    };
    worker.onerror = () => { stopTraining(); $("train-status").textContent = "Training failed"; notice("Worker failed. Check that you opened the game through npm start."); };
    worker.postMessage({ game: "ctf", episodes: budget, model, resetWeights: $("fresh").checked, rewardConfig, checkpointSelection: $("policy-selection").value });
  } catch (error) { stopTraining(); notice(error.message); }
};
$("epochs").onchange = () => { if (!worker) $("epoch").textContent = `0 / ${Number($("epochs").value).toLocaleString()}`; };
$("export").onclick = () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(model, null, 2)], { type: "application/json" }));
  const a = document.createElement("a"); a.href = url; a.download = "robot-capture-model.json"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("import").onclick = () => $("file").click();
$("file").onchange = async () => {
  try {
    const file = $("file").files[0];
    if (!file) return;
    if (file.size > 8000000) throw new Error("Import limit: 8,000,000 bytes.");
    const data = JSON.parse(await file.text());
    Network.load(data, CTF_SPEC);
    if (worker) { stopTraining(); $("train-status").textContent = "Stopped for model import"; }
    install(data); notice("Robot model imported. Next-run rewards remain as shown in the reward editor; saved run rewards appear with the checkpoint.");
  } catch (error) { notice("Could not import: " + error.message); }
  $("file").value = "";
};
function refreshEditor() {
  $("hidden-sizes").value = network.shape.slice(1, -1).join(", ");
  $("model-json").value = JSON.stringify(model, null, 2);
}
$("inspect").onclick = () => { refreshEditor(); $("inspector").showModal(); };
$("close-inspector").onclick = () => $("inspector").close();
$("rebuild").onclick = () => {
  try {
    const sizes = $("hidden-sizes").value.split(",").map((n) => Number(n.trim()));
    if (sizes.length < 1 || sizes.length > 4 || sizes.some((n) => !Number.isInteger(n) || n < 1 || n > 128)) throw new Error("Use 1–4 comma-separated hidden-layer sizes, each an integer from 1 to 128.");
    const next = new Network(null, Math.random, [96, ...sizes, 24], CTF_SPEC);
    if (worker) { stopTraining(); $("train-status").textContent = "Stopped for architecture rebuild"; }
    install({ ...next.serialize(), untrained: true }); refreshEditor();
    $("editor-status").textContent = "Architecture applied. All weights were randomized; previous evaluation metadata was cleared. Train this new structure or export it.";
  } catch (error) { $("editor-status").textContent = error.message; }
};
$("apply-model").onclick = () => {
  try {
    const next = Network.load(JSON.parse($("model-json").value), CTF_SPEC).serialize();
    if (worker) { stopTraining(); $("train-status").textContent = "Stopped for manual edits"; }
    install({ ...next, method: "manual-experiment" }); refreshEditor();
    $("editor-status").textContent = "Applied to the live controller and saved in this browser. Old evaluation metadata was cleared.";
  } catch (error) { $("editor-status").textContent = "Could not apply: " + error.message; }
};
showMap(); sync(); requestAnimationFrame(frame);

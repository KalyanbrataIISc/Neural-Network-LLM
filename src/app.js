import { brainPrompt, parseBrainJSON } from "./brain-prompt.js";
import { BrainStore } from "./brains.js";
import { createGame, GAMES, schedule } from "./games.js";
import { NetworkMap } from "./network-map.js";
import { W, H, R, PADDLE_W, PADDLE_Y, inputs } from "./game.js";
import {
  Network,
  ACTIONS,
  INPUT_NAMES,
  neuronNames,
  SHAPE,
} from "./network.js";
import { ACTION_FRAMES } from "./trainer.js";
const $ = (id) => document.getElementById(id),
  ctx = $("game").getContext("2d"),
  chart = $("chart").getContext("2d");
let game = createGame("breakout");
const brains = new BrainStore(localStorage);
let activeBrain = localStorage.getItem("active-brain"),
  nativeJob = null,
  nativePoll = null;
const STORAGE = "breakout-lab-dqn-v2";
let mode = "human",
  model = null,
  network = null,
  worker = null,
  history = [],
  keys = new Set(),
  target = null,
  last = 0,
  accumulator = 0,
  noticeTimer,
  draft = null,
  jsonDirty = false,
  aiAction = 1,
  aiTicks = 0,
  readySince = 0,
  endedSince = 0,
  manualSession = false,
  stepBusy = false,
  latestUpdates = "",
  trainingStarting = false;
const map = new NetworkMap($("network-map"), {
  getNetwork: () => network,
  commit: (next) => {
    if (worker || nativeJob) {
      stopTraining();
      notice("Training stopped to preserve your live edit.");
    }
    install(next.serialize());
  },
  notify: (message) => notice(message),
});
$("settings-open").onclick = () => $("settings-panel").showModal();
$("settings-close").onclick = () => $("settings-panel").close();
$("step-layer").onclick = () => {
  $("speed").value = "0";
  if (network) {
    const done = map.step(inputs(game.ball, game.paddle));
    if (done) {
      if (game.state === "ready") game.launch();
      if (game.state === "paused") game.state = "playing";
      const action = network.action(inputs(game.ball, game.paddle));
      for (let i = 0; i < ACTION_FRAMES; i++) game.step(1, ACTIONS[action]);
      sync();
    }
  }
};
function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => ($("notice").hidden = true), 6000);
}
function install(data, persist = true) {
  network = Network.load(data);
  model = data;
  if (persist) {
    try {
      localStorage.setItem(STORAGE, JSON.stringify(data));
    } catch {
      notice(
        "Model loaded, but browser storage is unavailable. Export weights to keep them.",
      );
    }
  }
  aiTicks = 0;
  map.reset();
  const count = network.weights.flat(2).length;
  $("experiment-shape").textContent = network.shape.join(" → ");
  const architecture = document.querySelector(".architecture");
  architecture.replaceChildren();
  network.shape.forEach((size, i) => {
    if (i) {
      const arrow = document.createElement("em");
      arrow.textContent = "→";
      architecture.append(arrow);
    }
    const column = document.createElement("div"),
      countLabel = document.createElement("b"),
      kind = document.createElement("span");
    countLabel.textContent = size;
    kind.textContent =
      i === 0
        ? "INPUTS"
        : i === network.shape.length - 1
          ? "ACTIONS"
          : `HIDDEN ${i}`;
    column.append(countLabel, kind);
    architecture.append(column);
  });
  $("parameter-count").textContent = `${count} parameters`;
  $("model-title").textContent = "Your paddle has a brain.";
  $("model-info").textContent =
    `${count} parameters${Number.isFinite(data.score) ? ` · ${data.score.toFixed(0)} evaluation score` : ""}. Weights loaded and ready to play.`;
  $("model-dot").style.background = "var(--mint)";
  $("deploy").disabled = false;
  $("export").disabled = false;
  $("inspect").disabled = false;
  if (!brains.records.length && !localStorage.getItem("brains-seeded")) {
    try {
      const b = brains.save("Starter · Breakout", data);
      activateBrain(b.id);
      localStorage.setItem("brains-seeded", "1");
    } catch (e) {
      notice(e.message);
    }
  }
}
try {
  const saved = localStorage.getItem(STORAGE);
  if (saved) install(JSON.parse(saved), false);
} catch {
  notice("Saved weights could not be loaded. Train or import a model.");
}
if (!model)
  fetch("/models/pretrained.json")
    .then((r) => {
      if (!r.ok) throw new Error("No bundled model");
      return r.json();
    })
    .then((data) => {
      if (!model && !worker) install(data, false);
    })
    .catch(() => {});
function setMode(next) {
  if (next === "ai" && !network) {
    notice("Train a network or import saved weights first.");
    return;
  }
  mode = next;
  keys.clear();
  target = null;
  aiTicks = 0;
  for (const id of ["human", "ai"]) {
    $(id).classList.toggle("selected", mode === id);
    $(id).setAttribute("aria-pressed", String(mode === id));
  }
  $("live-label").innerHTML =
    `<i></i> ${mode === "ai" ? "NEURAL NETWORK" : "HUMAN"} IN CONTROL`;
  sync();
}
$("human").onclick = () => setMode("human");
$("ai").onclick = () => setMode("ai");
$("deploy").onclick = () => {
  setMode("ai");
  game.reset();
  game.launch();
  sync();
  $("game").focus();
};
function toggle() {
  if (game.state === "won" || game.state === "lost") game.reset();
  game.launch();
  sync();
}
$("play").onclick = () => {
  toggle();
  $("game").focus();
};
$("pause").onclick = toggle;
$("reset").onclick = () => {
  game.reset();
  sync();
};
window.addEventListener("keydown", (e) => {
  if (e.defaultPrevented || e.target.closest("select,input,textarea,button,a"))
    return;
  if (["ArrowLeft", "ArrowRight", "a", "d", "A", "D", " "].includes(e.key)) {
    e.preventDefault();
    keys.add(e.key.toLowerCase());
    if (e.key === " " && !e.repeat) toggle();
  }
});
window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
window.addEventListener("blur", () => {
  keys.clear();
  if (game.state === "playing") {
    game.state = "paused";
    sync();
  }
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && game.state === "playing") {
    game.state = "paused";
    sync();
  }
});
$("game").addEventListener("pointermove", (e) => {
  if (mode === "human" && e.pointerType !== "mouse") {
    const rect = $("game").getBoundingClientRect();
    target = ((e.clientX - rect.left) / rect.width) * W;
  }
});
$("game").addEventListener("pointerdown", (e) => {
  if (e.pointerType !== "mouse") {
    e.preventDefault();
    $("game").setPointerCapture(e.pointerId);
    if (game.state !== "playing") toggle();
    const rect = $("game").getBoundingClientRect();
    target = ((e.clientX - rect.left) / rect.width) * W;
  }
});
$("game").addEventListener("pointerup", () => (target = null));
function sync() {
  $("score").textContent = String(game.score).padStart(4, "0");
  $("bricks").textContent =
    game.id === "catch" ? game.hits : game.bricks.filter((b) => b.alive).length;
  $("lives").textContent = "● ".repeat(game.lives).trim() || "—";
  $("pause").textContent =
    game.state === "playing"
      ? "Ⅱ Pause"
      : game.state === "paused"
        ? "▷ Resume"
        : "▷ Launch";
  $("overlay").hidden = game.state === "playing";
  const copy = {
    ready: [
      mode === "human" ? "YOU’RE IN CONTROL" : "NETWORK IN CONTROL",
      game.lives === 3 ? "Meet your paddle." : "Another chance.",
      mode === "human"
        ? "Move with ← → or A / D. Keep the ball in play."
        : "The saved network will move the paddle. Watch it play.",
      "Start playing",
    ],
    paused: [
      "TAKE A BREATHER",
      "Game paused.",
      "Your paddle will be right here.",
      "Resume game",
    ],
    won: [
      "ALL CLEAR",
      "Nicely played.",
      "Every brick down. Ready for another round?",
      "Play again",
    ],
    lost: [
      "EXPERIMENT COMPLETE",
      "One more round?",
      "Three lives, a little wiser. Give it another go.",
      "Try again",
    ],
  }[game.state];
  if (game.state === "ready" && mode === "ai")
    copy[2] = "Next serve launches automatically in a moment.";
  if (copy) {
    $("overlay-tag").textContent = copy[0];
    $("overlay-title").textContent = copy[1];
    $("overlay-copy").textContent = copy[2];
    $("play").textContent = copy[3] + " ↗";
  }
}
function rounded(x, y, w, h, r, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fill();
}
function draw() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#a5b89a10";
  for (let x = 20; x < W; x += 24)
    for (let y = 18; y < H; y += 24) ctx.fillRect(x, y, 1, 1);
  const colors = ["#bbef8b", "#9bcf73", "#78aa69", "#5e8968", "#426a5b"];
  for (const b of game.bricks)
    if (b.alive) {
      rounded(b.x, b.y, b.w, b.h, 3, colors[b.row]);
      ctx.fillStyle = "#ffffff22";
      ctx.fillRect(b.x + 3, b.y + 2, b.w - 6, 1);
    }
  for (let i = 0; i < game.trail.length; i++) {
    const p = game.trail[i];
    ctx.beginPath();
    ctx.arc(p.x, p.y, (R * i) / game.trail.length, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(197,244,157,${(i / game.trail.length) * 0.2})`;
    ctx.fill();
  }
  ctx.shadowColor = "#42bbae";
  ctx.shadowBlur = 16;
  rounded(game.paddle - PADDLE_W / 2, PADDLE_Y, PADDLE_W, 11, 5, "#166f65");
  ctx.beginPath();
  ctx.arc(game.ball.x, game.ball.y, R, 0, Math.PI * 2);
  ctx.fillStyle = "#166f65";
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = "#354231";
  ctx.beginPath();
  ctx.moveTo(24, H - 16);
  ctx.lineTo(W - 24, H - 16);
  ctx.stroke();
}
function frame(t) {
  const elapsed = Math.min((t - last) / 1000, 0.08);
  last = t;
  accumulator += elapsed * Number($("speed").value);
  if (
    ["won", "lost"].includes(game.state) &&
    $("auto-restart").checked &&
    Number($("speed").value) > 0
  ) {
    if (!endedSince) endedSince = t;
    if (t - endedSince > 1200) {
      game.reset();
      game.launch();
      endedSince = 0;
      sync();
    }
  } else endedSince = 0;
  if (mode === "ai" && game.state === "ready") {
    if (!readySince) readySince = t;
    if (t - readySince > 1000) {
      game.launch();
      ((readySince = 0),
        (endedSince = 0),
        (manualSession = false),
        (stepBusy = false),
        (latestUpdates = ""));
      sync();
    }
  } else readySince = 0;
  while (accumulator >= 1 / 120) {
    const prev = game.state;
    let movement =
      (keys.has("arrowright") || keys.has("d") ? 1 : 0) -
      (keys.has("arrowleft") || keys.has("a") ? 1 : 0);
    if (mode === "ai" && network) {
      if (aiTicks <= 0) {
        aiAction = network.action(inputs(game.ball, game.paddle));
        map.capture(network);
        aiTicks = ACTION_FRAMES * 2;
      }
      aiTicks--;
      movement = ACTIONS[aiAction];
    }
    game.step(0.5, movement, mode === "human" ? target : null);
    accumulator -= 1 / 120;
    if (prev !== game.state) {
      aiTicks = 0;
      sync();
    }
  }
  $("score").textContent = String(game.score).padStart(4, "0");
  $("bricks").textContent =
    game.id === "catch" ? game.hits : game.bricks.filter((b) => b.alive).length;
  if (mode === "human" && network && Number($("speed").value) > 0) {
    network.predict(inputs(game.ball, game.paddle));
    map.capture(network);
  }
  draw();
  map.draw(t, Number($("speed").value));
  requestAnimationFrame(frame);
}
function drawChart() {
  const w = 620,
    h = 130;
  chart.clearRect(0, 0, w, h);
  chart.strokeStyle = "#354132";
  chart.lineWidth = 1;
  for (let y = 15; y < h; y += 36) {
    chart.beginPath();
    chart.moveTo(0, y);
    chart.lineTo(w, y);
    chart.stroke();
  }
  if (history.length < 2) return;
  const max = Math.max(1, ...history) * 1.1;
  chart.beginPath();
  history.forEach((v, i) => {
    const x = (i / (history.length - 1)) * w,
      y = h - 8 - (v / max) * (h - 16);
    i ? chart.lineTo(x, y) : chart.moveTo(x, y);
  });
  chart.strokeStyle = "#42bbae";
  chart.lineWidth = 2;
  chart.stroke();
}
function stopTraining() {
  manualSession = false;
  stepBusy = false;
  worker?.terminate();
  if (nativeJob) {
    fetch(`/api/native/jobs/${nativeJob}`, { method: "DELETE" }).catch(
      () => {},
    );
    nativeJob = null;
    clearInterval(nativePoll);
  }
  worker = null;
  map.endTraining();
  $("train").innerHTML = "✧ <span>Train neural network</span><span>↗</span>";
  $("epochs").disabled = false;
  $("fresh").disabled = false;
}
$("epochs").onchange = () => {
  if (!worker)
    $("epoch").textContent =
      `0 / ${Number($("epochs").value).toLocaleString()}`;
};
function beginTraining(manual = false) {
  if (trainingStarting) return;
  if (worker || nativeJob) {
    stopTraining();
    $("train-status").textContent = "Training stopped";
    return;
  }
  let plan;
  try {
    plan = trainingPlan();
  } catch (e) {
    notice(e.message);
    return;
  }
  if ($("training-engine").value.startsWith("native")) {
    startNative(plan, manual);
    return;
  }
  manualSession = manual;
  stepBusy = manual;
  history = [];
  drawChart();
  $("epochs").disabled = true;
  $("fresh").disabled = true;
  $("train").innerHTML = "□ <span>Stop training</span>";
  $("train-status").textContent = "Starting training…";
  try {
    worker = new Worker(new URL("./train-worker.js", import.meta.url), {
      type: "module",
    });
    worker.onmessage = ({ data }) => {
      if (data.type === "progress") {
        if (data.snapshot) map.training(data.snapshot);
        if (data.byGame)
          $("per-game-eval").textContent = Object.entries(data.byGame)
            .map(([id, v]) => `${GAMES[id].name}: ${v.score.toFixed(0)} pts`)
            .join(" · ");
        if (
          manualSession ||
          data.episode % 100 === 0 ||
          data.episode === data.episodes
        ) {
          history.push(data.score);
          drawChart();
        }
        $("train-status").textContent = "Simulating at full speed";
        $("epoch").textContent =
          `${data.episode.toLocaleString()} / ${data.episodes.toLocaleString()}`;
        $("progress").style.width = `${(data.episode / data.episodes) * 100}%`;
        $("loss").innerHTML = `${data.score.toFixed(0)} <small>pts</small>`;
        $("samples").textContent =
          `${data.frames.toLocaleString()} simulated frames`;
        $("throughput").textContent =
          `${data.framesPerSecond.toLocaleString()} frames / sec`;
        $("exploration").textContent =
          `Exploration: ${Math.round(data.epsilon * 100)}%`;
      }
      if (data.replaySize !== undefined)
        latestUpdates = `Replay ${data.replaySize} / ${data.warmup} warmup · ${data.updates} optimizer updates`;
      if (data.type === "manual-step" || data.type === "manual-paused") {
        showParameterChanges(model, data.model);
        install(data.model);
        map.training({
          ...data.model,
          visualizationInput: data.visualizationInput || [0, 0, 0, 0, 0],
        });
        stepBusy = false;
        manualSession = true;
        setMode("ai");
        if (game.state === "ready") game.launch();
        sync();
        $("train-status").textContent =
          "Manual · episode complete · ready for Forward";
      }
      if (data.type === "checkpoint") {
        install(data.model);
      }
      if (data.type === "complete") {
        try {
          install(data.model);
          $("train-status").textContent = "Training complete · weights saved";
          notice(
            "Training complete. Select Watch AI play to try your network.",
          );
        } catch (e) {
          notice(e.message);
        }
        stopTraining();
      }
      if (data.type === "error") {
        notice(data.message);
        stopTraining();
        $("train-status").textContent = "Training failed. Please retry.";
      }
    };
    worker.onerror = () => {
      stopTraining();
      $("train-status").textContent = "Training failed";
      notice(
        "Could not start training. Run the app with npm start and open localhost:3000.",
      );
    };
    worker.postMessage({
      command: manual ? "step" : "run",
      episodes: Number($("epochs").value),
      plan,
      order: $("arena-order").value,
      gameId: game.id,
      model,
      resetWeights: $("fresh").checked,
    });
  } catch (e) {
    stopTraining();
    notice(e.message);
  }
}
$("train").onclick = () => beginTraining($("train-mode").value === "manual");
$("export").onclick = () => {
  if (!model) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(model, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "breakout-weights.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("import").onclick = () => $("file").click();
$("file").onchange = async () => {
  const file = $("file").files[0];
  if (!file) return;
  try {
    if (file.size > 1500000) throw new Error("Weight file is too large.");
    const data = JSON.parse(await file.text());
    const entry = brains.save(
      data.brainName || file.name.replace(/\.json$/i, ""),
      data,
    );
    activateBrain(entry.id);
    renderBrains();
    install(data);
    notice(
      "Brain imported and saved. Weights imported. Your network is ready to play.",
    );
  } catch (e) {
    notice("Could not import weights: " + e.message);
  }
  $("file").value = "";
};
sync();
drawChart();
requestAnimationFrame(frame);

function inspectNetwork() {
  if ($("inspect-layer").options.length !== draft.layers.length) {
    $("inspect-layer").replaceChildren(
      ...draft.layers.map(
        (l, i) =>
          new Option(`${l.name} · ${l.neurons.length} neurons`, String(i)),
      ),
    );
  }
  const layer = Number($("inspect-layer").value),
    data = draft.layers[layer],
    names = layer === 0 ? INPUT_NAMES : neuronNames(layer - 1, draft.shape);
  $("structure-summary").textContent =
    `${draft.shape.join(" → ")} · ${draft.layers.reduce((n, l) => n + l.neurons.reduce((s, v) => s + Object.keys(v.weights).length + 1, 0), 0)} parameters`;
  $("add-neuron").disabled =
    layer === draft.layers.length - 1 || draft.shape[layer + 1] >= 64;
  $("remove-neuron").disabled =
    layer === draft.layers.length - 1 || draft.shape[layer + 1] <= 1;
  $("neuron-equation").textContent =
    layer < draft.layers.length - 1
      ? "Each neuron = tanh(sum(input × weight) + bias)"
      : "Each Q value = sum(input × weight) + bias. The largest value selects the action.";
  const table = $("weight-table");
  table.replaceChildren();
  const head = document.createElement("tr");
  for (const name of ["Neuron", "Bias", ...names]) {
    const cell = document.createElement("th");
    cell.textContent = name;
    head.append(cell);
  }
  table.append(head);
  for (const neuron of data.neurons) {
    const row = document.createElement("tr"),
      label = document.createElement("td");
    label.textContent = neuron.name;
    row.append(label);
    for (const name of ["bias", ...names]) {
      const cell = document.createElement("td"),
        input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.value = name === "bias" ? neuron.bias : neuron.weights[name];
      input.setAttribute("aria-label", `${neuron.name} ${name}`);
      input.oninput = () => {
        const value = input.value === "" ? NaN : Number(input.value);
        if (name === "bias") neuron.bias = value;
        else neuron.weights[name] = value;
        jsonDirty = false;
        $("model-json").value = JSON.stringify(draft, null, 2);
        $("editor-status").textContent =
          "Draft changed. Apply edits to run this network.";
      };
      cell.append(input);
      row.append(cell);
    }
    table.append(row);
  }
  $("model-json").value = JSON.stringify(draft, null, 2);
  jsonDirty = false;
}
$("inspect").onclick = () => {
  draft = network ? network.serialize() : new Network().serialize();
  inspectNetwork();
  $("inspector").showModal();
};
$("inspect-layer").onchange = () => {
  if (jsonDirty) {
    try {
      draft = Network.load(JSON.parse($("model-json").value)).serialize();
    } catch (e) {
      $("editor-status").textContent = e.message;
      return;
    }
  }
  inspectNetwork();
};
$("close-inspector").onclick = () => $("inspector").close();
$("model-json").oninput = () => {
  jsonDirty = true;
  $("editor-status").textContent =
    "JSON draft changed. Apply edits to validate and run it.";
};
function currentOutputLayer() {
  return draft.layers.length - 1;
}
function changeNeurons(add) {
  try {
    const layer = Number($("inspect-layer").value);
    if (layer === currentOutputLayer()) return;
    const current = Network.load(
        jsonDirty ? JSON.parse($("model-json").value) : draft,
      ),
      shape = [...current.shape],
      weights = structuredClone(current.weights);
    if (add) {
      if (shape[layer + 1] >= 64)
        throw new Error(
          "This experiment supports up to 64 neurons per hidden layer.",
        );
      weights[layer].push(
        Array.from(
          { length: shape[layer] + 1 },
          () => Math.random() * 0.2 - 0.1,
        ),
      );
      for (const row of weights[layer + 1]) row.splice(row.length - 1, 0, 0);
      shape[layer + 1]++;
    } else {
      if (shape[layer + 1] <= 1)
        throw new Error("Keep at least one neuron in each hidden layer.");
      weights[layer].pop();
      for (const row of weights[layer + 1]) row.splice(shape[layer + 1] - 1, 1);
      shape[layer + 1]--;
    }
    draft = new Network(weights, Math.random, shape).serialize();
    inspectNetwork();
    $("editor-status").textContent = add
      ? "Neuron added. Outgoing connections start at zero; edit them to see its effect, or train the new structure."
      : "Last neuron removed from this layer. Apply to see the effect.";
  } catch (e) {
    $("editor-status").textContent = e.message;
  }
}
$("add-neuron").onclick = () => changeNeurons(true);
$("remove-neuron").onclick = () => changeNeurons(false);
$("apply-model").onclick = () => {
  try {
    const next = Network.load(
      jsonDirty ? JSON.parse($("model-json").value) : draft,
    ).serialize();
    if (worker || nativeJob) {
      stopTraining();
      $("train-status").textContent = "Training stopped for manual edits";
    }
    install({
      ...next,
      editedAt: new Date().toISOString(),
      method: "manual-experiment",
    });
    draft = next;
    inspectNetwork();
    $("editor-status").textContent =
      "Applied. The live game is now using your edited network. Weights saved in this browser.";
  } catch (e) {
    $("editor-status").textContent = "Could not apply: " + e.message;
  }
};
$("reload-model").onclick = async () => {
  try {
    const r = await fetch("/models/pretrained.json");
    if (!r.ok) throw new Error("Project weights not found.");
    const data = await r.json();
    Network.load(data);
    if (worker || nativeJob) stopTraining();
    install(data);
    draft = network.serialize();
    inspectNetwork();
    $("editor-status").textContent =
      "Project file reloaded and applied to the game.";
  } catch (e) {
    $("editor-status").textContent = e.message;
  }
};

const divider = document.createElement("div");
divider.className = "resize-divider";
divider.tabIndex = 0;
divider.setAttribute("role", "separator");
divider.setAttribute("aria-label", "Resize game and neural network views");
divider.setAttribute("aria-orientation", "vertical");
const workspace = document.querySelector(".workspace");
workspace.insertBefore(divider, document.querySelector(".network-panel"));
function setSplit(value) {
  const percent = Math.max(18, Math.min(65, value));
  workspace.style.setProperty("--game-width", percent + "%");
  divider.setAttribute("aria-valuenow", Math.round(percent));
  localStorage.setItem("breakout-split", percent);
}
try {
  setSplit(Number(localStorage.getItem("breakout-split") || 25));
} catch {}
divider.onpointerdown = (e) => {
  divider.setPointerCapture(e.pointerId);
  divider.onpointermove = (e) => {
    const r = workspace.getBoundingClientRect();
    setSplit(((e.clientX - r.left) / r.width) * 100);
  };
};
divider.onpointerup = () => (divider.onpointermove = null);
divider.onkeydown = (e) => {
  if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
    e.preventDefault();
    setSplit(
      Number(divider.getAttribute("aria-valuenow")) +
        (e.key === "ArrowRight" ? 2 : -2),
    );
  }
};

$("game-select").onchange = () => {
  game = createGame($("game-select").value);
  $("object-label").textContent = game.id === "catch" ? "CAUGHT" : "BRICKS";
  $("brick-total").hidden = game.id === "catch";
  sync();
  notice(`Same brain, new task: ${GAMES[game.id].name}.`);
};
function activateBrain(id) {
  activeBrain = id;
  if (id) localStorage.setItem("active-brain", id);
  else localStorage.removeItem("active-brain");
}
function renderBrains() {
  const list = $("brain-list");
  list.replaceChildren();
  $("brain-update").disabled =
    !activeBrain || !brains.records.some((b) => b.id === activeBrain);
  for (const brain of brains.records) {
    const card = document.createElement("article");
    card.className = "brain-card";
    const name = document.createElement("input");
    name.value = brain.name;
    name.maxLength = 80;
    name.setAttribute("aria-label", "Rename " + brain.name);
    name.onchange = () => {
      try {
        const renamed = brains.rename(brain.id, name.value);
        brain.name = renamed.name;
        $("brain-status").textContent = "Renamed.";
      } catch (e) {
        $("brain-status").textContent = e.message;
      }
    };
    const info = document.createElement("p");
    info.textContent = `${brain.model.shape.join(" → ")} · ${brain.id === activeBrain ? "Loaded · " : ""}${new Date(brain.updatedAt).toLocaleString()} · ${brain.model.observationSchema || "paddle-tracking-v1"}`;
    card.append(name, info);
    for (const [label, action] of [
      [
        "Load / edit",
        () => {
          if (worker || nativeJob) stopTraining();
          install(brain.model);
          activateBrain(brain.id);
          $("brains-panel").close();
          renderBrains();
          notice(
            "Brain loaded. Edit it on the live map; save updates in My Brains.",
          );
        },
      ],
      [
        "Export JSON",
        () =>
          downloadBrain({ ...brain.model, brainName: brain.name }, brain.name),
      ],
      [
        "Delete",
        () => {
          brains.delete(brain.id);
          if (activeBrain === brain.id) activateBrain(null);
          renderBrains();
          $("brain-status").textContent =
            "Deleted from library. The current live model remains available.";
        },
      ],
    ]) {
      const b = document.createElement("button");
      b.textContent = label;
      b.onclick = () => {
        try {
          action();
        } catch (e) {
          $("brain-status").textContent = e.message;
        }
      };
      card.append(b);
    }
    list.append(card);
  }
}
function downloadBrain(data, name) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name.replace(/[^a-z0-9_-]/gi, "_") + ".json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("brains-open").onclick = () => {
  renderBrains();
  $("brains-panel").showModal();
};
$("brains-close").onclick = () => $("brains-panel").close();
$("brain-save").onclick = () => {
  try {
    if (!model) throw new Error("Load a network first.");
    const b = brains.save($("brain-name").value, model);
    activateBrain(b.id);
    renderBrains();
    $("brain-status").textContent = "Brain saved.";
  } catch (e) {
    $("brain-status").textContent = e.message;
  }
};
$("brain-update").onclick = () => {
  try {
    const b = brains.get(activeBrain);
    brains.save(b.name, model, b.id);
    renderBrains();
    $("brain-status").textContent =
      "Loaded brain updated with the live weights.";
  } catch (e) {
    $("brain-status").textContent = e.message;
  }
};
$("brain-import").onclick = () => $("file").click();
function trainingPlan() {
  const plan = $("arena-enabled").checked
    ? Object.keys(GAMES)
        .map((id) => ({ game: id, trials: Number($("trials-" + id).value) }))
        .filter((r) => r.trials !== 0)
    : [{ game: game.id, trials: Number($("epochs").value) }];
  schedule(plan, $("arena-order").value);
  return plan;
}
async function nativeRequest(path, options) {
  const r = await fetch("/api/native" + path, options);
  const data = await r.json();
  if (!r.ok)
    throw new Error(
      data.error || "Native service unavailable. See README setup.",
    );
  return data;
}
$("check-device").onclick = async () => {
  try {
    const data = await nativeRequest("/status");
    $("device-status").textContent =
      `Native device: ${data.device} · ${data.name}`;
  } catch (e) {
    $("device-status").textContent = e.message;
  }
};
async function startNative(plan, manual = false) {
  trainingStarting = true;
  manualSession = manual;
  stepBusy = manual;
  history = [];
  drawChart();
  $("train").disabled = true;
  try {
    const data = await nativeRequest("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        mode: manual ? "manual" : "automatic",
        device: $("training-engine").value === "native-cpu" ? "cpu" : "auto",
        plan,
        order: $("arena-order").value,
        resetWeights: $("fresh").checked,
      }),
    });
    nativeJob = data.id;
    $("train").disabled = false;
    let checkpointEpisode = -1,
      lastStep = 0,
      lastEvaluation = -1;
    $("train").textContent = "Stop training";
    $("train-status").textContent = "Native training · " + data.device;
    $("device-status").textContent = "Training on " + data.device;
    nativePoll = setInterval(async () => {
      try {
        if (!nativeJob) return;
        const id = nativeJob;
        const state = await nativeRequest("/jobs/" + id);
        if (nativeJob !== id) return;
        if (state.checkpoint && state.checkpointEpisode !== checkpointEpisode) {
          install(state.checkpoint);
          checkpointEpisode = state.checkpointEpisode;
        }
        if (state.manualModel && state.manualModel.episodes !== lastStep) {
          showParameterChanges(model, state.manualModel);
          install(state.manualModel);
          lastStep = state.manualModel.episodes;
          stepBusy = false;
          setMode("ai");
          if (game.state === "ready") game.launch();
          sync();
        }
        if (state.snapshot) map.training(state.snapshot);
        if (state.framesPerSecond !== undefined)
          $("throughput").textContent =
            `${state.framesPerSecond.toLocaleString()} frames / sec`;
        latestUpdates = `${state.device} · batch ${state.profile?.batchSize || "—"} · replay ${state.replaySize || 0} · ${state.updates || 0} optimizer updates`;
        if (state.phase === "Waiting for Forward")
          $("train-status").textContent = "Manual · ready for Forward";
        if (state.frames)
          $("samples").textContent =
            `${state.frames.toLocaleString()} simulated frames`;
        $("epoch").textContent =
          `${state.episode || 0} / ${plan.reduce((s, r) => s + r.trials, 0)}`;
        $("progress").style.width =
          `${((state.episode || 0) / plan.reduce((s, r) => s + r.trials, 0)) * 100}%`;
        if (state.byGame && state.evaluationEpisode !== lastEvaluation) {
          lastEvaluation = state.evaluationEpisode;
          const values = Object.values(state.byGame);
          const mean = values.reduce((s, v) => s + v.score, 0) / values.length;
          $("loss").innerHTML = `${mean.toFixed(0)} <small>pts</small>`;
          history.push(mean);
          drawChart();
        }
        if (state.byGame)
          $("per-game-eval").textContent = Object.entries(state.byGame)
            .map(([id, v]) => `${GAMES[id].name}: ${v.score.toFixed(0)} pts`)
            .join(" · ");
        if (state.status === "complete") {
          install(state.model);
          nativeJob = null;
          clearInterval(nativePoll);
          stopTraining();
          $("train-status").textContent =
            "Native training complete · weights saved";
          notice(
            "Native training complete. Save your experiment in My Brains.",
          );
        }
        if (state.status === "error") throw new Error(state.error);
      } catch (e) {
        stopTraining();
        notice(e.message);
      }
    }, 1000);
  } catch (e) {
    stepBusy = false;
    manualSession = false;
    $("train-status").textContent = "Native training could not start";
    notice(e.message);
  } finally {
    trainingStarting = false;
    $("train").disabled = false;
  }
}

let builderWidths = [4];
function builderShape() {
  if (
    !builderWidths.length ||
    builderWidths.length > 6 ||
    builderWidths.some((n) => !Number.isInteger(n) || n < 1 || n > 64)
  )
    throw new Error("Choose 1–6 hidden layers with 1–64 neurons each.");
  return [5, ...builderWidths, 3];
}
function renderBuilder() {
  const list = $("builder-layers");
  list.replaceChildren();
  builderWidths.forEach((width, i) => {
    const row = document.createElement("div");
    row.className = "builder-row";
    const label = document.createElement("label");
    label.textContent = `Hidden layer ${i + 1}`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = 1;
    input.max = 64;
    input.step = 1;
    input.value = width;
    input.setAttribute("aria-label", `Hidden layer ${i + 1} neuron count`);
    input.oninput = () => (builderWidths[i] = Number(input.value));
    label.append(input);
    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.disabled = builderWidths.length === 1;
    remove.onclick = () => {
      builderWidths.splice(i, 1);
      renderBuilder();
    };
    row.append(label, remove);
    list.append(row);
  });
  $("builder-add-layer").disabled = builderWidths.length >= 6;
}
function showCreate() {
  if ($("brains-panel").open) $("brains-panel").close();
  $("create-status").textContent = "";
  renderBuilder();
  $("prompt-architecture").value = "current";
  promptWidths = [...(network?.shape.slice(1, -1) || [24, 24])];
  renderPromptControls();
  $("brain-create").showModal();
}
$("brain-add").onclick = showCreate;
$("create-close").onclick = () => $("brain-create").close();
$("builder-add-layer").onclick = () => {
  builderWidths.push(4);
  renderBuilder();
};
function generateDraft() {
  const next = new Network(
    null,
    $("builder-init").value === "zero" ? () => 0.5 : Math.random,
    builderShape(),
  ).serialize();
  $("create-json").value = JSON.stringify(next, null, 2);
  $("create-status").textContent =
    "Draft generated. You can edit every number below, save it, or continue on the map.";
  return next;
}
$("builder-generate").onclick = () => {
  try {
    generateDraft();
  } catch (e) {
    $("create-status").textContent = e.message;
  }
};
function saveCreated(data) {
  Network.load(data);
  const entry = brains.save($("create-name").value, data);
  if (worker || nativeJob) stopTraining();
  install(data);
  activateBrain(entry.id);
  renderBrains();
  $("brain-create").close();
  notice(
    "Brain saved and loaded. Drag neurons/connections or right-click to edit live; save changes in My Brains.",
  );
}
$("builder-map").onclick = () => {
  try {
    saveCreated(generateDraft());
  } catch (e) {
    $("create-status").textContent = e.message;
  }
};
$("create-validate").onclick = () => {
  try {
    const data = parseBrainJSON($("create-json").value),
      net = Network.load(data);
    $("create-status").textContent =
      `Valid brain: ${net.shape.join(" → ")} · ${net.weights.flat(2).length} parameters. Nothing was executed or trained during validation.`;
  } catch (e) {
    $("create-status").textContent = "Invalid brain: " + e.message;
  }
};
$("create-save").onclick = () => {
  try {
    saveCreated(parseBrainJSON($("create-json").value));
  } catch (e) {
    $("create-status").textContent = "Could not save: " + e.message;
  }
};
$("create-import").onclick = () => $("create-file").click();
const createFile = document.createElement("input");
createFile.id = "create-file";
createFile.type = "file";
createFile.accept = ".json,application/json";
createFile.hidden = true;
$("brain-create").append(createFile);
createFile.onchange = async () => {
  const file = createFile.files[0];
  if (!file) return;
  try {
    if (file.size > 1500000) throw new Error("File too large.");
    const text = await file.text();
    const data = parseBrainJSON(text);
    Network.load(data);
    $("create-json").value = JSON.stringify(data, null, 2);
    $("create-name").value =
      data.brainName || file.name.replace(/\.json$/i, "");
    $("create-status").textContent =
      "Valid imported draft. Review its name and save.";
  } catch (e) {
    $("create-status").textContent = e.message;
  }
  createFile.value = "";
};
async function preparePrompt() {
  const shape = promptShape();
  const selectedGame = game.id;
  $("create-status").textContent =
    "Preparing the complete prompt from the selected game source…";
  const [shared, adapters] = await Promise.all([
    fetch("/src/game.js").then((r) => {
      if (!r.ok) throw new Error("Could not load game source.");
      return r.text();
    }),
    fetch("/src/games.js").then((r) => {
      if (!r.ok) throw new Error("Could not load task source.");
      return r.text();
    }),
  ]);
  const text = brainPrompt({
    gameId: selectedGame,
    shape,
    sharedSource: shared,
    gameSource: adapters,
  });
  $("llm-prompt").value = text;
  return text;
}
$("prompt-show").onclick = async () => {
  try {
    await preparePrompt();
    $("prompt-details").open = true;
    $("create-status").textContent =
      "Prompt ready. You can copy the full text below.";
  } catch (e) {
    $("create-status").textContent = e.message;
  }
};
$("prompt-copy").onclick = async () => {
  try {
    const text = await preparePrompt();
    try {
      await navigator.clipboard.writeText(text);
      $("create-status").textContent =
        "Prompt copied. Paste into any LLM, then paste its complete JSON response here.";
    } catch {
      $("prompt-details").open = true;
      $("llm-prompt").focus();
      $("llm-prompt").select();
      $("create-status").textContent =
        "Clipboard unavailable. The prompt is selected below; use Cmd/Ctrl+C.";
    }
  } catch (e) {
    $("create-status").textContent = e.message;
  }
};

$("create-current").onclick = () => {
  if (!model) {
    $("create-status").textContent = "Load a brain first.";
    return;
  }
  $("create-json").value = JSON.stringify(model, null, 2);
  $("create-name").value =
    (activeBrain
      ? brains.records.find((b) => b.id === activeBrain)?.name ||
        "Current brain"
      : "Current brain") + " copy";
  $("create-status").textContent =
    "Current brain copied into an editable draft. Saving creates a separate named brain.";
};

let promptWidths = [24, 24];
function promptShape() {
  const shape =
    $("prompt-architecture").value === "current"
      ? [...(network?.shape || [5, 24, 24, 3])]
      : [5, ...promptWidths, 3];
  if (
    shape.length < 3 ||
    shape.length > 8 ||
    shape.slice(1, -1).some((n) => !Number.isInteger(n) || n < 1 || n > 64)
  )
    throw new Error(
      "Choose 1–6 hidden layers with 1–64 neurons each for the prompt.",
    );
  return shape;
}
function invalidatePrompt() {
  $("llm-prompt").value = "";
  $("create-status").textContent =
    "Prompt architecture changed. View or copy to generate the updated prompt.";
}
function renderPromptControls() {
  const custom = $("prompt-architecture").value === "custom";
  $("prompt-custom").hidden = !custom;
  const list = $("prompt-layers");
  list.replaceChildren();
  promptWidths.forEach((width, i) => {
    const row = document.createElement("div");
    row.className = "builder-row";
    const label = document.createElement("label");
    label.textContent = `Prompt hidden layer ${i + 1}`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = 1;
    input.max = 64;
    input.step = 1;
    input.value = width;
    input.setAttribute(
      "aria-label",
      `Prompt hidden layer ${i + 1} neuron count`,
    );
    input.oninput = () => {
      promptWidths[i] = Number(input.value);
      invalidatePrompt();
      updatePromptSummary();
    };
    label.append(input);
    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.disabled = promptWidths.length === 1;
    remove.onclick = () => {
      promptWidths.splice(i, 1);
      invalidatePrompt();
      renderPromptControls();
    };
    row.append(label, remove);
    list.append(row);
  });
  $("prompt-add-layer").disabled = promptWidths.length >= 6;
  updatePromptSummary();
}
function updatePromptSummary() {
  try {
    const shape = promptShape();
    $("prompt-shape-summary").textContent =
      `${shape.join(" → ")} · ${shape.length - 2} hidden layer(s) · ${shape.slice(1).reduce((sum, n, i) => sum + n * (shape[i] + 1), 0)} parameters. Independent of the scratch builder.`;
  } catch (e) {
    $("prompt-shape-summary").textContent = e.message;
  }
}
$("prompt-architecture").onchange = () => {
  if ($("prompt-architecture").value === "custom")
    promptWidths = [...(network?.shape.slice(1, -1) || [24, 24])];
  invalidatePrompt();
  renderPromptControls();
};
$("prompt-add-layer").onclick = () => {
  promptWidths.push(promptWidths.at(-1) || 4);
  invalidatePrompt();
  renderPromptControls();
};
$("prompt-reset-current").onclick = () => {
  promptWidths = [...(network?.shape.slice(1, -1) || [24, 24])];
  invalidatePrompt();
  renderPromptControls();
};

$("auto-restart").checked = localStorage.getItem("auto-restart") !== "false";
$("auto-restart").onchange = () =>
  localStorage.setItem("auto-restart", $("auto-restart").checked);
$("builder-zero-default").onclick = () => {
  const zero = new Network(null, () => 0.5, SHAPE).serialize();
  $("create-json").value = JSON.stringify(zero, null, 2);
  $("create-name").value = "All-zero brain";
  $("create-status").textContent =
    "Exact zero draft: default layers and all 819 parameters are zero. Save to load. Hidden zero weights can remain symmetric during training; random Reset weights is optional.";
};
function showParameterChanges(previous, next) {
  const before = previous ? Network.load(previous).weights.flat(2) : [],
    after = Network.load(next).weights.flat(2);
  let changed = 0,
    max = 0;
  after.forEach((v, i) => {
    const delta = Math.abs(v - (before[i] || 0));
    if (delta > 1e-12) {
      changed++;
      max = Math.max(max, delta);
    }
  });
  $("parameter-changes").textContent =
    `Displayed brain update: ${changed} / ${after.length} parameters changed · largest |Δ| ${max.toExponential(3)}. Exact zero hidden weights can stay zero through symmetry.`;
}
$("training-settings").onclick = () => $("settings-panel").showModal();
$("training-stop").onclick = () => {
  stopTraining();
  $("train-status").textContent = "Training stopped";
};
$("main-train").onclick = () => {
  if (worker || nativeJob) {
    if (manualSession) {
      manualSession = false;
      $("train-mode").value = "automatic";
      if (worker) worker.postMessage({ command: "run" });
      else
        nativeRequest(`/jobs/${nativeJob}/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        }).catch((e) => notice(e.message));
      $("train-status").textContent = "Running at maximum speed";
    } else {
      if (worker) worker.postMessage({ command: "pause" });
      else
        nativeRequest(`/jobs/${nativeJob}/pause`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        }).catch((e) => notice(e.message));
      manualSession = true;
      $("train-mode").value = "manual";
    }
  } else beginTraining($("train-mode").value === "manual");
};
$("training-forward").onclick = () => {
  if (stepBusy) return;
  if (worker || nativeJob) {
    if (!manualSession) {
      $("main-train").click();
      return;
    }
    stepBusy = true;
    if (worker) worker.postMessage({ command: "step" });
    else
      nativeRequest(`/jobs/${nativeJob}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      }).catch((e) => {
        stepBusy = false;
        notice(e.message);
      });
  } else {
    $("train-mode").value = "manual";
    beginTraining(true);
  }
};
setInterval(() => {
  $("monitor-status").textContent = $("train-status").textContent;
  $("monitor-trials").textContent = $("epoch").textContent;
  $("monitor-speed").textContent = $("throughput").textContent;
  $("monitor-score").textContent = "Evaluation: " + $("loss").textContent;
  $("monitor-details").textContent = [
    $("per-game-eval").textContent,
    $("samples").textContent,
    latestUpdates,
  ]
    .filter(Boolean)
    .join(" · ");
  $("main-chart").getContext("2d").clearRect(0, 0, 400, 70);
  $("main-chart").getContext("2d").drawImage($("chart"), 0, 0, 400, 70);
  $("main-train").textContent =
    worker || nativeJob
      ? manualSession
        ? "Run at maximum speed"
        : "Pause for manual stepping"
      : $("train-mode").value === "manual"
        ? "Start manual training"
        : "Run at maximum speed";
  $("training-forward").disabled = stepBusy || trainingStarting;
  $("main-train").disabled = trainingStarting;
  $("training-stop").disabled = trainingStarting;
}, 250);

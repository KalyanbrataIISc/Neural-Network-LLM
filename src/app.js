import { Game, W, H, R, PADDLE_W, PADDLE_Y, inputs } from "./game.js";
import { Network, ACTIONS, INPUT_NAMES, neuronNames } from "./network.js";
import { ACTION_FRAMES } from "./trainer.js";
const $ = (id) => document.getElementById(id),
  game = new Game(),
  ctx = $("game").getContext("2d"),
  chart = $("chart").getContext("2d");
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
  readySince = 0;
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
  const count = network.weights.flat(2).length;
  $("experiment-shape").textContent =
    `${network.shape[0]} INPUTS → ${network.shape[1]} → ${network.shape[2]} → ${network.shape[3]} ACTIONS`;
  $("hidden-1-count").textContent = network.shape[1];
  $("hidden-2-count").textContent = network.shape[2];
  $("parameter-count").textContent = `${count} parameters`;
  $("model-title").textContent = "Your paddle has a brain.";
  $("model-info").textContent =
    `${count} parameters${Number.isFinite(data.score) ? ` · ${data.score.toFixed(0)} evaluation score` : ""}. Weights loaded and ready to play.`;
  $("model-dot").style.background = "var(--mint)";
  $("deploy").disabled = false;
  $("export").disabled = false;
  $("inspect").disabled = false;
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
  if (e.target.closest("select,input,button,a")) return;
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
  $("bricks").textContent = game.bricks.filter((b) => b.alive).length;
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
  ctx.shadowColor = "#b9f577";
  ctx.shadowBlur = 16;
  rounded(game.paddle - PADDLE_W / 2, PADDLE_Y, PADDLE_W, 11, 5, "#c6f397");
  ctx.beginPath();
  ctx.arc(game.ball.x, game.ball.y, R, 0, Math.PI * 2);
  ctx.fillStyle = "#f1f7df";
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
  accumulator += elapsed;
  if (mode === "ai" && game.state === "ready") {
    if (!readySince) readySince = t;
    if (t - readySince > 1000) {
      game.launch();
      readySince = 0;
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
  $("bricks").textContent = game.bricks.filter((b) => b.alive).length;
  draw();
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
  chart.strokeStyle = "#b9f577";
  chart.lineWidth = 2;
  chart.stroke();
}
function stopTraining() {
  worker?.terminate();
  worker = null;
  $("train").innerHTML = "✧ <span>Train neural network</span><span>↗</span>";
  $("epochs").disabled = false;
  $("fresh").disabled = false;
}
$("epochs").onchange = () => {
  if (!worker)
    $("epoch").textContent =
      `0 / ${Number($("epochs").value).toLocaleString()}`;
};
$("train").onclick = () => {
  if (worker) {
    stopTraining();
    $("train-status").textContent = "Training stopped";
    return;
  }
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
        if (data.episode % 100 === 0 || data.episode === data.episodes) {
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
      episodes: Number($("epochs").value),
      model,
      resetWeights: $("fresh").checked,
    });
  } catch (e) {
    stopTraining();
    notice(e.message);
  }
};
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
    install(data);
    notice("Weights imported. Your network is ready to play.");
  } catch (e) {
    notice("Could not import weights: " + e.message);
  }
  $("file").value = "";
};
sync();
drawChart();
requestAnimationFrame(frame);

function inspectNetwork() {
  const layer = Number($("inspect-layer").value),
    data = draft.layers[layer],
    names = layer === 0 ? INPUT_NAMES : neuronNames(layer - 1, draft.shape);
  $("structure-summary").textContent =
    `${draft.shape.join(" → ")} · ${draft.layers.reduce((n, l) => n + l.neurons.reduce((s, v) => s + Object.keys(v.weights).length + 1, 0), 0)} parameters`;
  $("add-neuron").disabled = layer === 2 || draft.shape[layer + 1] >= 64;
  $("remove-neuron").disabled = layer === 2 || draft.shape[layer + 1] <= 1;
  $("neuron-equation").textContent =
    layer < 2
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
function changeNeurons(add) {
  try {
    const layer = Number($("inspect-layer").value);
    if (layer === 2) return;
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
    if (worker) {
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
    if (worker) stopTraining();
    install(data);
    draft = network.serialize();
    inspectNetwork();
    $("editor-status").textContent =
      "Project file reloaded and applied to the game.";
  } catch (e) {
    $("editor-status").textContent = e.message;
  }
};

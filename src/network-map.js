import { OBSERVATION_NAMES } from "./games.js";
import { Network, OUTPUT_NAMES } from "./network.js";
const $ = (id) => document.getElementById(id);
const color = (v, alpha = 1) =>
  v >= 0 ? `rgba(0,132,151,${alpha})` : `rgba(208,87,67,${alpha})`;
// One clock drives packet position and layer changes. A packet reaches 1.0
// before the next stage begins; game inference cannot replace a trace mid-cycle.
export function flowClock(milliseconds, duration = 1200, layers = 4) {
  const stage = Math.floor(milliseconds / duration) % layers;
  return {
    phase: stage,
    progress: (milliseconds % duration) / duration,
    cycle: Math.floor(milliseconds / (duration * layers)),
  };
}
export class NetworkMap {
  constructor(canvas, { getNetwork, commit, notify }) {
    Object.assign(this, { canvas, getNetwork, commit, notify });
    this.ctx = canvas.getContext("2d");
    this.nodes = [];
    this.edges = [];
    this.positions = {};
    this.phase = 0;
    this.trace = null;
    this.pendingTrace = null;
    this.pendingTrain = null;
    this.cycle = 0;
    this.edgeProgress = 0;
    this.traceWeights = null;
    this.selected = null;
    this.hover = null;
    this.trainNet = null;
    this.lastPhase = 0;
    this.motion = 0;
    this.previousTime = 0;
    this.stepping = false;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.oncontextmenu = (e) => {
      e.preventDefault();
      this.openEditor(this.hit(e), e);
    };
    canvas.onpointerdown = (e) => {
      if (e.button !== 0) return;
      const h = this.hit(e);
      if (h?.type === "node") {
        this.drag = { node: h, move: e.shiftKey };
        canvas.setPointerCapture(e.pointerId);
      }
      this.selected = h;
    };
    canvas.onpointermove = (e) => {
      this.pointer = this.point(e);
      if (this.drag?.move) {
        this.positions[this.drag.node.id] = this.pointer;
      }
      this.hover = this.hit(e);
      const tip = $("graph-tooltip");
      tip.hidden = !this.hover;
      if (this.hover) {
        tip.textContent = this.describe(this.hover);
        tip.style.left = Math.min(this.pointer.x + 12, this.width - 240) + "px";
        tip.style.top = Math.max(0, this.pointer.y - 32) + "px";
      }
    };
    canvas.onpointerup = (e) => {
      if (this.drag && !this.drag.move) {
        const end = this.hit(e),
          start = this.drag.node;
        if (end?.type === "node" && end.layer === start.layer + 1) {
          this.editNetwork((net) => {
            net.weights[start.layer][end.index][start.index] = 0.1;
          });
        } else if (end?.type === "node" && end.id !== start.id)
          this.notify(
            "Connect a neuron to the next layer to keep this network feed-forward.",
          );
      }
      this.drag = null;
    };
    canvas.ondragover = (e) => e.preventDefault();
    canvas.ondrop = (e) => {
      e.preventDefault();
      const p = this.point(e),
        layer = Math.max(
          1,
          Math.min(
            (this.getNetwork()?.shape.length || 4) - 2,
            Math.round(
              ((p.x / this.width - 0.08) / 0.84) *
                ((this.getNetwork()?.shape.length || 4) - 1),
            ),
          ),
        );
      this.addNeuron(layer, p);
    };
    $("neuron-palette").ondragstart = (e) =>
      e.dataTransfer.setData("text/plain", "neuron");
    $("neuron-palette").onkeydown = (e) => {
      if (e.key === "Enter") this.addNeuron(1);
    };
    $("graph-edit-close").onclick = () => ($("graph-edit").hidden = true);
    $("graph-value").oninput = () => this.applyValue();
    $("graph-value").onkeydown = (e) => {
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        $("graph-value").value = (
          Number($("graph-value").value) + (e.key === "ArrowUp" ? 0.01 : -0.01)
        ).toFixed(6);
        this.applyValue();
      }
      if (e.key === "Escape") $("graph-edit").hidden = true;
    };
    $("graph-zero").onclick = () => {
      $("graph-value").value = 0;
      this.applyValue();
    };
  }
  resize() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    this.width = r.width;
    this.height = r.height;
    const d = devicePixelRatio || 1;
    this.canvas.width = r.width * d;
    this.canvas.height = r.height * d;
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
  }
  reset() {
    this.trace = null;
    this.trainNet = null;
    this.phase = 0;
    this.motion = 0;
    this.cycle = 0;
    this.stepping = false;
    this.pendingTrace = null;
    this.pendingTrain = null;
    this.traceWeights = null;
  }
  capture(net) {
    if (this.trainNet && net !== this.trainNet) return;
    const trace = net.activations.map((a) => Array.from(a));
    if (!this.trace) {
      this.trace = trace;
      this.traceWeights = structuredClone(net.weights);
    } else this.pendingTrace = trace;
  }
  training(data) {
    const next = Network.load(data);
    next.predict(data.visualizationInput || [0, 0, 0, 0, 0]);
    if (!this.trainNet) {
      this.trainNet = next;
      this.pendingTrace = null;
      this.trace = next.activations.map((a) => Array.from(a));
      this.traceWeights = structuredClone(next.weights);
      this.motion = 0;
      this.cycle = 0;
    } else this.pendingTrain = next;
  }
  endTraining() {
    this.reset();
  }
  step(input) {
    const net = this.getNetwork();
    if (!net) return false;
    if (!this.stepping || this.phase === net.shape.length - 1) {
      this.stepping = true;
      net.predict(input);
      this.trace = net.activations.map((a) => Array.from(a));
      this.traceWeights = structuredClone(net.weights);
      this.phase = 0;
      this.edgeProgress = 1;
      return false;
    }
    this.phase++;
    this.edgeProgress = 1;
    return this.phase === net.shape.length - 1;
  }
  point(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  hit(e) {
    const p = this.point(e);
    const near = this.nodes
      .map((n) => ({ n, d: Math.hypot(n.x - p.x, n.y - p.y) }))
      .sort((a, b) => a.d - b.d)[0];
    if (near && near.d < 12) return near.n;
    let best = null,
      distance = 6;
    for (const edge of this.edges) {
      const { a, b } = edge,
        dx = b.x - a.x,
        dy = b.y - a.y,
        t = Math.max(
          0,
          Math.min(
            1,
            ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy),
          ),
        ),
        d = Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
      if (d < distance) {
        distance = d;
        best = edge;
      }
    }
    return best;
  }
  describe(item) {
    const net = this.trainNet || this.getNetwork();
    if (item.type === "edge")
      return `${item.a.name} → ${item.b.name} · w ${net.weights[item.a.layer][item.b.index][item.a.index].toFixed(5)}`;
    return `${item.name} · activation ${(this.trace?.[item.layer]?.[item.index] || 0).toFixed(4)}${item.layer ? " · bias " + net.weights[item.layer - 1][item.index].at(-1).toFixed(4) : ""}`;
  }
  editNetwork(fn) {
    try {
      const source = this.trainNet || this.getNetwork();
      if (!source) return;
      const net = new Network(
        structuredClone(source.weights),
        Math.random,
        source.shape,
      );
      fn(net);
      this.commit(Network.load(net.serialize()));
      this.trace = null;
    } catch (e) {
      this.notify(e.message);
    }
  }
  addNeuron(layer, position) {
    this.editNetwork((net) => {
      if (net.shape[layer] >= 64)
        throw new Error("Maximum 64 neurons per hidden layer.");
      net.weights[layer - 1].push(
        Array.from(
          { length: net.shape[layer - 1] + 1 },
          () => Math.random() * 0.2 - 0.1,
        ),
      );
      for (const row of net.weights[layer]) row.splice(row.length - 1, 0, 0);
      net.shape[layer]++;
      const next = new Network(net.weights, Math.random, net.shape);
      Object.assign(net, next);
      if (position)
        this.positions[`${layer}:${net.shape[layer] - 1}`] = position;
    });
    this.notify(
      "Neuron added. Drag a connection to the next layer; new outgoing weights start at zero.",
    );
  }
  openEditor(item, e) {
    if (!item) return;
    if (item.type === "node" && item.layer === 0) {
      this.notify(
        "Input neurons receive the game state; edit an outgoing connection instead.",
      );
      return;
    }
    this.editing = item;
    const net = this.trainNet || this.getNetwork();
    $("graph-edit-label").textContent =
      item.type === "edge"
        ? `${item.a.name} → ${item.b.name}`
        : `${item.name} bias`;
    $("graph-value").value =
      item.type === "edge"
        ? net.weights[item.a.layer][item.b.index][item.a.index]
        : net.weights[item.layer - 1][item.index].at(-1);
    $("graph-edit").hidden = false;
    $("graph-value").focus();
    $("graph-value").select();
  }
  applyValue() {
    const value = Number($("graph-value").value);
    if (
      $("graph-value").value === "" ||
      !Number.isFinite(value) ||
      Math.abs(value) > 1e6
    )
      return;
    const item = this.editing;
    this.editNetwork((net) => {
      if (item.type === "edge")
        net.weights[item.a.layer][item.b.index][item.a.index] = value;
      else
        net.weights[item.layer - 1][item.index][net.shape[item.layer - 1]] =
          value;
    });
  }
  draw(t, speed) {
    let net = this.trainNet || this.getNetwork();
    const c = this.ctx,
      w = this.width,
      h = this.height;
    if (!w || !h) return;
    this.motion += Math.min(80, t - this.previousTime) * speed;
    this.previousTime = t;
    c.clearRect(0, 0, w, h);
    if (!net) return;
    if (!this.trace) {
      net.predict([0, 0, 0, 0, 0]);
      this.capture(net);
    }
    if (speed > 0) {
      this.stepping = false;
      const clock = flowClock(this.motion, 1200, net.shape.length);
      this.phase = clock.phase;
      this.edgeProgress = clock.progress;
      if (clock.cycle !== this.cycle) {
        this.cycle = clock.cycle;
        if (this.pendingTrain) {
          this.trainNet = this.pendingTrain;
          net = this.trainNet;
          this.trace = net.activations.map((a) => Array.from(a));
          this.pendingTrain = null;
        } else if (this.pendingTrace) {
          this.trace = this.pendingTrace;
          this.pendingTrace = null;
        }
        this.traceWeights = structuredClone(net.weights);
      }
    }
    this.nodes = [];
    this.edges = [];
    const grid = net.shape.map(() => []);
    const columns = net.shape.map(
      (_, i) => 0.08 + (i / (net.shape.length - 1)) * 0.84,
    );
    for (let layer = 0; layer < net.shape.length; layer++) {
      const count = net.shape[layer];
      for (let i = 0; i < count; i++) {
        const id = `${layer}:${i}`,
          p = this.positions[id] || {
            x: columns[layer] * w,
            y: 45 + ((i + 0.5) / count) * (h - 90),
          };
        const current = {
          type: "node",
          id,
          layer,
          index: i,
          x: p.x,
          y: p.y,
          name:
            layer === 0
              ? OBSERVATION_NAMES[i]
              : layer === net.shape.length - 1
                ? OUTPUT_NAMES[i]
                : `h${layer}.${i + 1}`,
        };
        this.nodes.push(current);
        grid[layer][i] = current;
      }
    }
    const node = (l, i) => grid[l][i],
      sel = this.selected;
    for (let l = 0; l < net.weights.length; l++)
      for (let j = 0; j < net.shape[l + 1]; j++)
        for (let i = 0; i < net.shape[l]; i++) {
          const a = node(l, i),
            b = node(l + 1, j),
            value = (this.traceWeights || net.weights)[l][j][i],
            edge = { type: "edge", a, b };
          this.edges.push(edge);
          const relevant =
              sel?.type === "node" && (sel.id === a.id || sel.id === b.id),
            active = this.phase === l + 1,
            signal = Math.min(1, Math.abs((this.trace?.[l]?.[i] || 0) * value));
          c.strokeStyle = color(
            value,
            Math.min(
              0.72,
              0.04 + Math.abs(value) * 0.13 + (relevant ? 0.35 : 0),
            ),
          );
          c.lineWidth = 1;
          c.beginPath();
          c.moveTo(a.x, a.y);
          c.lineTo(b.x, b.y);
          c.stroke();
          const followed =
            sel?.type === "edge" && sel.a.id === a.id && sel.b.id === b.id;
          if (active && (signal > 0.15 || followed)) {
            const progress = this.edgeProgress;
            c.fillStyle = color(value, 0.8);
            c.shadowColor = color(value);
            c.shadowBlur = 6;
            c.beginPath();
            c.arc(
              a.x + (b.x - a.x) * progress,
              a.y + (b.y - a.y) * progress,
              1.5 + signal,
              0,
              Math.PI * 2,
            );
            c.fill();
            c.shadowBlur = 0;
            if (followed) {
              const x = a.x + (b.x - a.x) * progress,
                y = a.y + (b.y - a.y) * progress;
              c.fillStyle = "white";
              c.fillRect(x - 50, y - 22, 100, 16);
              c.fillStyle = color(value);
              c.font = "10px monospace";
              c.textAlign = "center";
              c.fillText(
                `${this.trace[l][i].toFixed(2)} × ${value.toFixed(2)}`,
                x,
                y - 10,
              );
            }
          }
          if (relevant && sel.id === b.id) {
            const x = a.x + (b.x - a.x) * 0.18,
              y = a.y + (b.y - a.y) * 0.18;
            c.fillStyle = "#fff";
            c.fillRect(x - 17, y - 6, 34, 12);
            c.fillStyle = color(value);
            c.font = "9px monospace";
            c.textAlign = "center";
            c.fillText(value.toFixed(2), x, y + 3);
          }
        }
    for (const n of this.nodes) {
      const v = this.trace?.[n.layer]?.[n.index] || 0,
        active =
          n.layer < this.phase ||
          (n.layer === this.phase &&
            (this.phase === 0 || this.edgeProgress > 0.94));
      c.shadowColor = color(v);
      c.shadowBlur = active ? Math.min(15, 4 + Math.abs(v) * 9) : 0;
      c.beginPath();
      c.arc(
        n.x,
        n.y,
        Math.min(10, (h - 90) / net.shape[n.layer] / 2.6),
        0,
        Math.PI * 2,
      );
      c.fillStyle = active
        ? color(v, 0.25 + Math.min(0.7, Math.abs(v) * 0.6))
        : "#fff";
      c.fill();
      c.strokeStyle = color(v, 0.8);
      c.lineWidth = sel?.id === n.id ? 3 : 1.4;
      c.stroke();
      c.shadowBlur = 0;
      if (n.layer === 0 || n.layer === net.shape.length - 1) {
        c.font = "10px monospace";
        c.textAlign = n.layer === 0 ? "left" : "right";
        c.fillStyle = "#36565e";
        c.fillText(
          n.name +
            (n.layer === net.shape.length - 1 &&
            n.index ===
              this.trace.at(-1).indexOf(Math.max(...this.trace.at(-1)))
              ? " ← chosen"
              : ""),
          n.layer === 0 ? 5 : w - 5,
          n.y - (n.layer === 0 ? 6 : 14),
        );
        c.fillText(
          v.toFixed(3),
          n.layer === 0 ? 5 : w - 5,
          n.y + (n.layer === 0 ? 8 : 17),
        );
      }
    }
    c.font = "10px system-ui";
    c.textAlign = "center";
    c.fillStyle = "#688087";
    net.shape
      .map((n, i) =>
        i === 0
          ? "INPUTS"
          : i === net.shape.length - 1
            ? "ACTIONS"
            : `HIDDEN ${i} · ${n}`,
      )
      .forEach((label, i) => c.fillText(label, w * columns[i], 20));
    if (this.drag && !this.drag.move && this.pointer) {
      c.strokeStyle = "#166f65";
      c.setLineDash([4, 4]);
      c.beginPath();
      c.moveTo(this.drag.node.x, this.drag.node.y);
      c.lineTo(this.pointer.x, this.pointer.y);
      c.stroke();
      c.setLineDash([]);
    }
    const focus = this.selected || this.hover,
      weights = this.traceWeights || net.weights;
    let explanation =
      "Select a connection to follow activation × weight, or a neuron to inspect its sum and bias.";
    if (
      focus?.type === "edge" &&
      weights[focus.a.layer]?.[focus.b.index]?.[focus.a.index] !== undefined
    ) {
      const input = this.trace[focus.a.layer][focus.a.index],
        weight = weights[focus.a.layer][focus.b.index][focus.a.index];
      explanation = `${focus.a.name} → ${focus.b.name}: ${input.toFixed(5)} × ${weight.toFixed(5)} = ${(input * weight).toFixed(5)} contribution. Travel ${(this.edgeProgress * 100).toFixed(0)}%.`;
    } else if (
      focus?.type === "node" &&
      focus.layer > 0 &&
      weights[focus.layer - 1]?.[focus.index]
    ) {
      const row = weights[focus.layer - 1][focus.index],
        bias = row.at(-1),
        sum = this.trace[focus.layer - 1].reduce(
          (total, v, i) => total + v * row[i],
          0,
        );
      explanation = `${focus.name}: Σ(a × w) ${sum.toFixed(5)} + bias ${bias.toFixed(5)} = ${(sum + bias).toFixed(5)} → ${focus.layer === net.shape.length - 1 ? "linear" : "tanh"} → ${this.trace[focus.layer][focus.index].toFixed(5)}`;
    }
    $("flow-detail").textContent = explanation;
    $("graph-stage").textContent =
      `${this.trainNet ? "TRAINING SNAPSHOT · replay sample" : "INFERENCE REPLAY · captured activations"} · ${this.phase === 0 ? "Read game state" : this.phase === net.shape.length - 1 ? "Choose highest Q value" : `Hidden layer ${this.phase} · tanh`}`;
  }
}

import { Network, INPUT_NAMES, OUTPUT_NAMES } from "./network.js";
const $ = (id) => document.getElementById(id);
const color = (v, alpha = 1) =>
  v >= 0 ? `rgba(0,132,151,${alpha})` : `rgba(208,87,67,${alpha})`;
export class NetworkMap {
  constructor(canvas, { getNetwork, commit, notify }) {
    Object.assign(this, { canvas, getNetwork, commit, notify });
    this.ctx = canvas.getContext("2d");
    this.nodes = [];
    this.edges = [];
    this.positions = {};
    this.phase = 0;
    this.trace = null;
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
        layer = p.x < this.width / 2 ? 1 : 2;
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
  }
  capture(net) {
    if (this.trainNet && net !== this.trainNet) return;
    this.trace = net.activations.map((a) => Array.from(a));
  }
  training(data) {
    this.trainNet = Network.load(data);
    this.trainNet.predict(data.visualizationInput || [0, 0, 0, 0, 0]);
    this.capture(this.trainNet);
  }
  endTraining() {
    this.trainNet = null;
    this.trace = null;
  }
  step(input) {
    const net = this.getNetwork();
    if (!net) return false;
    if (!this.stepping) {
      this.stepping = true;
      this.phase = 3;
    }
    if (!this.trace || this.phase === 3) {
      net.predict(input);
      this.capture(net);
      this.phase = 0;
      return false;
    }
    this.phase++;
    return this.phase === 3;
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
    const net = this.trainNet || this.getNetwork(),
      c = this.ctx,
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
    if (speed > 0) this.stepping = false;
    if (speed > 0 && t - this.lastPhase > Math.max(80, 220 / speed)) {
      this.phase = (this.phase + 1) % 4;
      this.lastPhase = t;
    }
    this.nodes = [];
    this.edges = [];
    const columns = [0.08, 0.35, 0.65, 0.92];
    for (let layer = 0; layer < 4; layer++) {
      const count = net.shape[layer];
      for (let i = 0; i < count; i++) {
        const id = `${layer}:${i}`,
          p = this.positions[id] || {
            x: columns[layer] * w,
            y: 45 + ((i + 0.5) / count) * (h - 90),
          };
        this.nodes.push({
          type: "node",
          id,
          layer,
          index: i,
          x: p.x,
          y: p.y,
          name:
            layer === 0
              ? INPUT_NAMES[i]
              : layer === 3
                ? OUTPUT_NAMES[i]
                : `h${layer}.${i + 1}`,
        });
      }
    }
    const node = (l, i) =>
        this.nodes.find((n) => n.layer === l && n.index === i),
      sel = this.selected;
    for (let l = 0; l < 3; l++)
      for (let j = 0; j < net.shape[l + 1]; j++)
        for (let i = 0; i < net.shape[l]; i++) {
          const a = node(l, i),
            b = node(l + 1, j),
            value = net.weights[l][j][i],
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
          if (active && signal > 0.15) {
            const progress =
              (this.motion % Math.max(200, 700 / Math.max(speed, 0.05))) /
              Math.max(200, 700 / Math.max(speed, 0.05));
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
        active = n.layer === this.phase;
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
      if (n.layer === 0 || n.layer === 3) {
        c.font = "10px monospace";
        c.textAlign = n.layer === 0 ? "left" : "right";
        c.fillStyle = "#36565e";
        c.fillText(
          n.name +
            (n.layer === 3 &&
            n.index === this.trace[3].indexOf(Math.max(...this.trace[3]))
              ? " ← chosen"
              : ""),
          n.layer === 0 ? 5 : w - 5,
          n.y - 16,
        );
        c.fillText(v.toFixed(3), n.layer === 0 ? 5 : w - 5, n.y + 24);
      }
    }
    c.font = "10px system-ui";
    c.textAlign = "center";
    c.fillStyle = "#688087";
    [
      "INPUTS",
      `HIDDEN 1 · ${net.shape[1]}`,
      `HIDDEN 2 · ${net.shape[2]}`,
      "ACTIONS",
    ].forEach((label, i) => c.fillText(label, w * columns[i], 20));
    if (this.drag && !this.drag.move && this.pointer) {
      c.strokeStyle = "#166f65";
      c.setLineDash([4, 4]);
      c.beginPath();
      c.moveTo(this.drag.node.x, this.drag.node.y);
      c.lineTo(this.pointer.x, this.pointer.y);
      c.stroke();
      c.setLineDash([]);
    }
    $("graph-stage").textContent =
      `${this.trainNet ? "TRAINING SNAPSHOT · replay sample" : "INFERENCE REPLAY · captured activations"} · ${["Read game state", "Hidden layer 1 · tanh", "Hidden layer 2 · tanh", "Choose highest Q value"][this.phase]}`;
  }
}

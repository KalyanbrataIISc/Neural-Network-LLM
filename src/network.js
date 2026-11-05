export const SHAPE = [5, 24, 24, 3];
export const ACTIONS = [-1, 0, 1];
export const INPUT_NAMES = [
  "ball_x",
  "ball_y",
  "velocity_x",
  "velocity_y",
  "paddle_x",
];
export const OUTPUT_NAMES = ["move_left", "stay", "move_right"];
export const LAYER_NAMES = ["hidden_1", "hidden_2", "actions"];
export function neuronNames(layer, shape = SHAPE) {
  return layer === 2
    ? OUTPUT_NAMES
    : Array.from(
        { length: shape[layer + 1] },
        (_, i) => `${LAYER_NAMES[layer]}_${String(i + 1).padStart(2, "0")}`,
      );
}

export function rng(seed = 42) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++)
    if (values[i] > values[best]) best = i;
  return best;
}
// Internal layout: weights[layer][neuron][input]. The last number in each
// neuron's row is its bias. serialize() exposes the same data with names so
// every connection can be edited in plain JSON without understanding indices.
export class Network {
  constructor(weights, random = Math.random, shape = SHAPE) {
    this.shape = [...shape];
    this.weights =
      weights ||
      this.shape
        .slice(1)
        .map((n, l) =>
          Array.from({ length: n }, () =>
            Array.from({ length: this.shape[l] + 1 }, (_, j) =>
              j === this.shape[l]
                ? 0
                : (random() * 2 - 1) * Math.sqrt(6 / (this.shape[l] + n)),
            ),
          ),
        );
    this.activations = this.shape.map((n) => new Float64Array(n));
    this.deltas = this.shape.map((n) => new Float64Array(n));
    this.gradients = this.weights.map((layer) =>
      layer.map((row) => new Float64Array(row.length)),
    );
    this.m = this.weights.map((layer) =>
      layer.map((row) => new Float64Array(row.length)),
    );
    this.v = this.weights.map((layer) =>
      layer.map((row) => new Float64Array(row.length)),
    );
    this.updates = 0;
  }
  predict(input) {
    this.activations[0].set(input);
    for (let l = 0; l < this.weights.length; l++) {
      const a = this.activations[l],
        out = this.activations[l + 1];
      for (let j = 0; j < out.length; j++) {
        const w = this.weights[l][j];
        let sum = w[a.length];
        for (let i = 0; i < a.length; i++) sum += w[i] * a[i];
        out[j] = l === 2 ? sum : Math.tanh(sum);
      }
    }
    return this.activations[3];
  }
  action(input) {
    return argmax(this.predict(input));
  }
  clearGradients() {
    for (const layer of this.gradients) for (const row of layer) row.fill(0);
  }
  // Accumulate a Huber-loss gradient for the action actually tried in a game.
  accumulate(input, action, target) {
    const pred = this.predict(input)[action],
      a = this.activations,
      d = this.deltas;
    const error = pred - target;
    d[3].fill(0);
    d[3][action] = Math.max(-1, Math.min(1, error));
    for (let l = 2; l > 0; l--)
      for (let i = 0; i < this.shape[l]; i++) {
        let sum = 0;
        for (let j = 0; j < this.shape[l + 1]; j++)
          sum += this.weights[l][j][i] * d[l + 1][j];
        d[l][i] = sum * (1 - a[l][i] * a[l][i]);
      }
    for (let l = 0; l < 3; l++)
      for (let j = 0; j < this.shape[l + 1]; j++) {
        const g = this.gradients[l][j];
        for (let i = 0; i < this.shape[l]; i++) g[i] += d[l + 1][j] * a[l][i];
        g[this.shape[l]] += d[l + 1][j];
      }
    return Math.abs(error) < 1 ? (error * error) / 2 : Math.abs(error) - 0.5;
  }
  // Adam averages a replay batch before updating each individual parameter.
  update(batchSize, rate = 0.0006) {
    this.updates++;
    const c1 = 1 - 0.9 ** this.updates,
      c2 = 1 - 0.999 ** this.updates;
    for (let l = 0; l < 3; l++)
      for (let j = 0; j < this.shape[l + 1]; j++)
        for (let i = 0; i <= this.shape[l]; i++) {
          const g = this.gradients[l][j][i] / batchSize,
            m = this.m[l][j],
            v = this.v[l][j];
          m[i] = 0.9 * m[i] + 0.1 * g;
          v[i] = 0.999 * v[i] + 0.001 * g * g;
          this.weights[l][j][i] -=
            (rate * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
        }
  }
  copyFrom(network) {
    for (let l = 0; l < 3; l++)
      for (let j = 0; j < this.shape[l + 1]; j++)
        for (let i = 0; i <= this.shape[l]; i++)
          this.weights[l][j][i] = network.weights[l][j][i];
  }
  serialize() {
    return {
      version: 2,
      algorithm: "double-dqn",
      shape: this.shape,
      inputs: INPUT_NAMES,
      outputs: OUTPUT_NAMES,
      explanation:
        "Hidden neurons: tanh(sum(input × weight) + bias). Action neurons: sum(input × weight) + bias. Choose the action with the highest Q value.",
      layers: this.weights.map((layer, l) => ({
        name: LAYER_NAMES[l],
        activation: l === 2 ? "linear" : "tanh",
        neurons: layer.map((row, j) => ({
          name: neuronNames(l, this.shape)[j],
          bias: row[this.shape[l]],
          weights: Object.fromEntries(
            (l === 0 ? INPUT_NAMES : neuronNames(l - 1, this.shape)).map(
              (name, i) => [name, row[i]],
            ),
          ),
        })),
      })),
    };
  }
  static load(data) {
    if (
      data?.version !== 2 ||
      data.algorithm !== "double-dqn" ||
      JSON.stringify(data.inputs) !== JSON.stringify(INPUT_NAMES) ||
      JSON.stringify(data.outputs) !== JSON.stringify(OUTPUT_NAMES) ||
      !Array.isArray(data.shape) ||
      data.shape.length !== 4 ||
      data.shape[0] !== 5 ||
      data.shape[3] !== 3 ||
      data.shape
        .slice(1, 3)
        .some((n) => !Number.isInteger(n) || n < 1 || n > 64) ||
      !Array.isArray(data.layers) ||
      data.layers.length !== 3
    )
      throw new Error(
        "Unsupported model format. Use a Breakout Lab reinforcement-learning checkpoint.",
      );
    const weights = data.layers.map((layer, l) => {
      if (
        layer.name !== LAYER_NAMES[l] ||
        layer.activation !== (l === 2 ? "linear" : "tanh") ||
        !Array.isArray(layer.neurons) ||
        layer.neurons.length !== data.shape[l + 1]
      )
        throw new Error("Invalid model layers.");
      return layer.neurons.map((neuron, j) => {
        const names = l === 0 ? INPUT_NAMES : neuronNames(l - 1, data.shape);
        if (
          neuron.name !== neuronNames(l, data.shape)[j] ||
          !neuron.weights ||
          Object.keys(neuron.weights).length !== names.length
        )
          throw new Error("Invalid neuron structure.");
        const row = [...names.map((name) => neuron.weights[name]), neuron.bias];
        if (row.some((v) => !Number.isFinite(v) || Math.abs(v) > 1000000))
          throw new Error("Invalid model weights.");
        return row;
      });
    });
    return new Network(weights, Math.random, data.shape);
  }
}

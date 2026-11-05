import test from "node:test";
import assert from "node:assert/strict";
import { Game, reflect, PADDLE_Y, R, inputs } from "../src/game.js";
import { Network, rng } from "../src/network.js";
import { transition } from "../src/trainer.js";

test("side-wall reflections handle multiple bounces", () => {
  assert.equal(reflect(903), 883);
  assert.equal(reflect(-3), 17);
  assert.equal(reflect(7 + 886 * 4 + 100), 107);
});
test("game physics handles paddle, brick and missed-ball collisions", () => {
  const g = new Game(rng());
  g.launch();
  g.ball = { x: 450, y: PADDLE_Y - R - 1, vx: 0, vy: 4 };
  g.step(1);
  assert.ok(g.ball.vy < 0);
  assert.equal(g.hits, 1);
  const b = g.bricks[40];
  g.ball = { x: b.x + 20, y: b.y + b.h + R + 1, vx: 0, vy: -4 };
  g.step(1);
  assert.equal(b.alive, false);
  assert.equal(g.score, 10);
  g.ball.y = 560;
  g.step(1);
  assert.equal(g.lives, 2);
  assert.equal(g.state, "ready");
});
test("named model roundtrip preserves predictions and rejects corrupt structure", () => {
  const n = new Network(null, rng()),
    a = [0.1, 0.2, 0.3, 0.4, 0.5],
    model = n.serialize();
  assert.ok(Object.hasOwn(model.layers[0].neurons[0].weights, "ball_x"));
  const loaded = Network.load(JSON.parse(JSON.stringify(model)));
  const predicted = [...n.predict(a)];
  [...loaded.predict(a)].forEach((v, i) =>
    assert.ok(Math.abs(v - predicted[i]) < 1e-7),
  );
  const bad = structuredClone(model);
  bad.layers[0].neurons[0].weights.ball_x = null;
  assert.throws(() => Network.load(bad));
  assert.throws(() => Network.load({ ...model, shape: [5, 999, 24, 3] }));
});
test("editing action biases changes the action actually executed", () => {
  const n = new Network(null, rng()),
    model = n.serialize();
  for (const neuron of model.layers[2].neurons) {
    for (const name in neuron.weights) neuron.weights[name] = 0;
    neuron.bias = neuron.name === "move_right" ? 10 : 0;
  }
  const edited = Network.load(model),
    g = new Game(rng());
  g.launch();
  const x = g.paddle;
  const action = edited.action(inputs(g.ball, g.paddle));
  assert.equal(action, 2);
  transition(g, action);
  assert.ok(g.paddle > x);
});
test("adding a neuron with zero outgoing weights preserves output and supports learning", () => {
  const n = new Network(null, rng()),
    w = structuredClone(n.weights),
    a = [0.1, 0.2, 0.3, 0.4, 0.5];
  w[0].push([0.2, 0.1, 0.3, 0.2, 0.1, 0]);
  for (const row of w[1]) row.splice(row.length - 1, 0, 0);
  const expanded = new Network(w, Math.random, [5, 25, 24, 3]);
  [...n.predict(a)].forEach((v, i) =>
    assert.ok(Math.abs(v - expanded.predict(a)[i]) < 1e-12),
  );
  const loaded = Network.load(expanded.serialize());
  assert.deepEqual(loaded.shape, [5, 25, 24, 3]);
  loaded.clearGradients();
  loaded.accumulate(a, 0, 2);
  loaded.update(1);
  assert.ok(Number.isFinite(loaded.predict(a)[0]));
});
test("DQN optimization reduces temporal-difference error for a sampled transition", () => {
  const n = new Network(null, rng()),
    input = [0, 0.5, 0.2, 0.7, 0.4],
    target = 2;
  const before = Math.abs(n.predict(input)[2] - target);
  for (let i = 0; i < 120; i++) {
    n.clearGradients();
    n.accumulate(input, 2, target);
    n.update(1);
  }
  assert.ok(Math.abs(n.predict(input)[2] - target) < before * 0.1);
});
test("missed balls end RL transitions; paddle hits earn a reward", () => {
  const g = new Game(rng());
  g.launch();
  g.ball = { x: 450, y: PADDLE_Y - R - 1, vx: 0, vy: 4 };
  assert.ok(transition(g, 1).reward > 1);
  g.ball = { x: 450, y: 545, vx: 0, vy: 4 };
  const result = transition(g, 1);
  assert.equal(result.done, true);
  assert.ok(result.reward < 0);
});

import test from "node:test";
import assert from "node:assert/strict";
import { BrainStore } from "../src/brains.js";
import { Network, rng } from "../src/network.js";
import { createGame, schedule, observe } from "../src/games.js";
import { train } from "../src/trainer.js";
import { flowClock } from "../src/network-map.js";
test("brain library validates, persists copies, renames and deletes", () => {
  const data = new Map(),
    storage = {
      getItem: (k) => data.get(k) || null,
      setItem: (k, v) => data.set(k, v),
    };
  const library = new BrainStore(storage),
    model = new Network(null, rng(1)).serialize();
  const entry = library.save("Experiment", model);
  model.layers[0].neurons[0].bias = 100;
  assert.notEqual(library.get(entry.id).model.layers[0].neurons[0].bias, 100);
  library.rename(entry.id, "Renamed");
  assert.equal(new BrainStore(storage).get(entry.id).name, "Renamed");
  assert.throws(() =>
    library.save("Bad", { ...model, observationSchema: "unknown" }),
  );
  assert.equal(library.records.length, 1);
  library.delete(entry.id);
  assert.equal(new BrainStore(storage).records.length, 0);
});
test("plans schedule exact trials in both orders", () => {
  const plan = [
    { game: "breakout", trials: 3 },
    { game: "catch", trials: 2 },
  ];
  assert.deepEqual(schedule(plan), [
    "breakout",
    "catch",
    "breakout",
    "catch",
    "breakout",
  ]);
  assert.deepEqual(schedule(plan, "sequential"), [
    "breakout",
    "breakout",
    "breakout",
    "catch",
    "catch",
  ]);
  assert.throws(() => schedule([{ game: "catch", trials: -1 }]));
  assert.throws(() => schedule([{ game: "other", trials: 1 }]));
});
test("Catch shares observations and actions without immediately winning on empty bricks", () => {
  const g = createGame("catch", rng());
  g.launch();
  g.ball = { x: g.paddle, y: 480, vx: 0, vy: 4 };
  g.step(1, 0);
  assert.equal(g.hits, 1);
  assert.equal(g.score, 10);
  assert.equal(g.state, "playing");
  assert.equal(observe(g).length, 5);
  g.ball = { x: 10, y: 546, vx: 0, vy: 4 };
  g.step(1, 0);
  assert.equal(g.lives, 2);
  assert.equal(g.state, "ready");
});
test("mixed training records exact completed counts and produces a reusable brain", () => {
  const initialModel = new Network(null, rng(), [5, 2, 3, 3]).serialize();
  const model = train({
    initialModel,
    plan: [
      { game: "breakout", trials: 2 },
      { game: "catch", trials: 1 },
    ],
    episodes: 3,
  });
  assert.deepEqual(model.completedTrials, { breakout: 2, catch: 1 });
  assert.deepEqual(Network.load(model).shape, [5, 2, 3, 3]);
  assert.equal(model.trainingHistory.length, 1);
});
test("one animation clock reaches each layer before the next edge starts", () => {
  const before = flowClock(2399),
    after = flowClock(2400);
  assert.equal(before.phase, 1);
  assert.ok(before.progress > 0.999);
  assert.equal(after.phase, 2);
  assert.equal(after.progress, 0);
  assert.equal(before.cycle, after.cycle);
  assert.equal(flowClock(4800).cycle, 1);
});

test("variable hidden layers roundtrip and train, including a one-layer handwritten brain", () => {
  for (const shape of [
    [5, 4, 3],
    [5, 2, 3, 2, 3],
  ]) {
    const net = new Network(null, rng(22), shape);
    const loaded = Network.load(net.serialize());
    assert.deepEqual(loaded.shape, shape);
    const sample = [0.1, 0.2, 0.3, 0.4, 0.5],
      expected = [...net.predict(sample)];
    assert.deepEqual([...loaded.predict(sample)], expected);
    const before = Math.abs(loaded.predict(sample)[1] - 2);
    for (let i = 0; i < 400; i++) {
      loaded.clearGradients();
      loaded.accumulate(sample, 1, 2);
      loaded.update(1);
    }
    assert.ok(Math.abs(loaded.predict(sample)[1] - 2) < before * 0.4);
    const result = train({
      initialModel: loaded.serialize(),
      plan: [{ game: "catch", trials: 1 }],
    });
    assert.deepEqual(Network.load(result).shape, shape);
  }
});

test("LLM prompt and pasted JSON have a complete valid handwritten contract", async () => {
  const { brainPrompt, parseBrainJSON } =
    await import("../src/brain-prompt.js");
  const prompt = brainPrompt({
    gameId: "catch",
    sharedSource: "SHARED PHYSICS",
    gameSource: "TASK PHYSICS",
  });
  assert.match(prompt, /Do NOT use tools, internet/);
  assert.match(prompt, /simulations/);
  assert.match(prompt, /SHARED PHYSICS/);
  assert.match(prompt, /TASK PHYSICS/);
  assert.match(prompt, /target.vy\/6/);
  const model = new Network(null, rng(), [5, 4, 3]).serialize();
  const wrapped = "```json\n" + JSON.stringify(model) + "\n```";
  assert.deepEqual(Network.load(parseBrainJSON(wrapped)).shape, [5, 4, 3]);
  assert.throws(() => parseBrainJSON("const brain = {}"));
});

test("prompt templates preserve the exact requested architecture", async () => {
  const { brainPrompt } = await import("../src/brain-prompt.js");
  for (const shape of [
    [5, 24, 24, 3],
    [5, 8, 6, 2, 3],
  ]) {
    const text = brainPrompt({ shape });
    const template = JSON.parse(
      text.match(
        /COMPLETE VALID STRUCTURE TO FILL[^\n]*\n([\s\S]*?)\n\nEXACT JAVASCRIPT/,
      )[1],
    );
    assert.deepEqual(Network.load(template).shape, shape);
    assert.equal(template.layers.length, shape.length - 1);
  }
});

test("manual training advances exactly one episode and retains replay and optimizer state", async () => {
  const { trainingSession } = await import("../src/trainer.js");
  const initialModel = new Network(null, () => 0.5).serialize(),
    progress = [];
  const session = trainingSession({
    episodes: 3,
    isManual: () => true,
    initialModel,
    onProgress: (p) => progress.push(p),
  });
  const first = session.next();
  assert.equal(first.value.episode, 1);
  assert.ok(progress.at(-1).updates > 0);
  assert.ok(first.value.model.layers.at(-1).neurons.some((n) => n.bias !== 0));
  assert.ok(
    first.value.model.layers[0].neurons.every(
      (n) => n.bias === 0 && Object.values(n.weights).every((w) => w === 0),
    ),
  );
  const replay = progress.at(-1).replaySize,
    updates = progress.at(-1).updates;
  const second = session.next();
  assert.equal(second.value.episode, 2);
  assert.ok(progress.at(-1).replaySize > replay);
  assert.ok(progress.at(-1).updates > updates);
});

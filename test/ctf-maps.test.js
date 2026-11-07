import test from "node:test";
import assert from "node:assert/strict";
import { generateMap, mapSplit, MAP_PROTOCOL, MAP_SPLITS, CELL_SIZE } from "../src/ctf-maps.js";
import { CaptureGame, CTF_ENV, CTF_SPEC, captureInputs, captureTransition, visible } from "../src/ctf-game.js";
import { Network, rng } from "../src/network.js";
import { train } from "../src/trainer.js";

test("the entire 10,000-map library is deterministic and has distinct geometry/spawns", () => {
  const fingerprints = new Set();
  for (let id = 1; id <= MAP_PROTOCOL.total; id++) {
    const map = generateMap(id);
    assert.equal(map.id, id);
    assert.equal(map.split, mapSplit(id));
    assert.equal(map.enemies.length, 4); assert.equal(map.supplies.length, 4);
    fingerprints.add(JSON.stringify([map.base, map.flag, map.walls, map.enemies, map.supplies]));
  }
  assert.equal(fingerprints.size, 10000);
  const original = JSON.stringify(generateMap(4237));
  for (let id = 1; id <= 200; id++) generateMap(id); // Evict cached map.
  assert.equal(JSON.stringify(generateMap(4237)), original);
  for (const invalid of [0.5, -1, 10001, NaN, "2"]) assert.throws(() => generateMap(invalid));
});

test("generated arenas have collision-clear spawns and a connected, short objective route", () => {
  const cols = 12, rows = 8;
  const point = (cell) => ({ x: (cell % cols + .5) * CELL_SIZE, y: (Math.floor(cell / cols) + .5) * CELL_SIZE });
  const obstructed = (p, walls) => walls.some((w) => p.x + 12 > w.x && p.x - 12 < w.x + w.w && p.y + 12 > w.y && p.y - 12 < w.y + w.h);
  for (let id = 1; id <= 10000; id += 97) {
    const map = generateMap(id);
    for (const p of [map.base, map.flag, ...map.enemies.map(([x, y]) => ({ x, y })), ...map.supplies.map(([x, y]) => ({ x, y }))])
      assert.equal(obstructed(p, map.walls), false, `Spawn clear on map ${id}`);
    const cellFor = (p) => Math.floor(p.y / CELL_SIZE) * cols + Math.floor(p.x / CELL_SIZE);
    const start = cellFor(map.base), target = cellFor(map.flag), queue = [start], costs = new Map([[start, 0]]);
    for (let index = 0; index < queue.length; index++) {
      const cell = queue[index], x = cell % cols, y = Math.floor(cell / cols);
      for (const next of [x > 0 ? cell - 1 : -1, x < cols - 1 ? cell + 1 : -1, y > 0 ? cell - cols : -1, y < rows - 1 ? cell + cols : -1]) {
        if (next < 0 || costs.has(next)) continue;
        const a = point(cell), b = point(next), midpoint = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
        if (obstructed(midpoint, map.walls)) continue;
        costs.set(next, costs.get(cell) + 1); queue.push(next);
      }
    }
    assert.equal(costs.size, 96, `All rooms connected on map ${id}`);
    assert.ok(costs.get(target) <= 18, `Short flag route on map ${id}`);
  }
});

test("training, validation and test map IDs are disjoint and episode factories enforce the split", () => {
  assert.equal(mapSplit(1), "training"); assert.equal(mapSplit(1000), "training");
  assert.equal(mapSplit(1001), "validation"); assert.equal(mapSplit(2000), "validation");
  assert.equal(mapSplit(2001), "test"); assert.equal(mapSplit(10000), "test");
  const random = rng(42);
  for (let i = 0; i < 200; i++) {
    const training = CTF_ENV.createEpisode(random, { purpose: "training" });
    assert.equal(training.map.split, "training");
    const evaluation = CTF_ENV.createEpisode(random, { purpose: "evaluation", evaluationIndex: i });
    assert.ok(MAP_PROTOCOL.validationMapIds.includes(evaluation.mapId));
    assert.equal(evaluation.map.split, "validation");
  }
  assert.equal(new CaptureGame().map.split, "test");
});

test("changing a map updates collisions, objective sensors and all spawn positions", () => {
  const game = new CaptureGame(rng(9), 1234), initialWalls = game.walls;
  game.state = "playing"; captureTransition(game, 18);
  game.reset(8765);
  const expected = generateMap(8765), values = captureInputs(game);
  assert.notDeepEqual(game.walls, initialWalls);
  assert.equal(game.mapId, 8765); assert.equal(game.state, "ready");
  assert.equal(game.player.x, expected.base.x); assert.equal(game.player.y, expected.base.y);
  assert.equal(values[10], (expected.flag.x - expected.base.x) / 960);
  assert.equal(values[11], (expected.flag.y - expected.base.y) / 640);
  assert.equal(values.length, 96);
  const wall = game.walls[0];
  const a = wall.h > wall.w ? { x: wall.x - 20, y: wall.y + wall.h / 2 } : { x: wall.x + wall.w / 2, y: wall.y - 20 };
  const b = wall.h > wall.w ? { x: wall.x + wall.w + 20, y: a.y } : { x: a.x, y: wall.y + wall.h + 20 };
  assert.equal(visible(a, b, game.walls), false);
  assert.equal(visible(a, b, []), true);
});

test("a training run records only training maps, while evaluation uses separate validation maps", () => {
  const observed = [];
  const environment = { ...CTF_ENV, createEpisode: (random, context) => {
    const game = CTF_ENV.createEpisode(random, context);
    observed.push({ purpose: context.purpose, id: game.mapId }); return game;
  } };
  const model = new Network(null, rng(4), [96, 2, 24], CTF_SPEC).serialize();
  const result = train({ episodes: 2, initialModel: model, environment, checkpointSelection: "latest" });
  assert.equal(result.mapProtocol.version, "ctf-maps-v1");
  assert.ok(result.trainingMapIds.length > 0);
  assert.ok(result.trainingMapIds.every((id) => id >= MAP_SPLITS.training.start && id <= MAP_SPLITS.training.end));
  assert.ok(observed.filter((row) => row.purpose === "evaluation").every((row) => mapSplit(row.id) === "validation"));
  assert.equal(observed.filter((row) => row.purpose === "training").length, 2);
  assert.ok(observed.every((row) => mapSplit(row.id) !== "test"));
});

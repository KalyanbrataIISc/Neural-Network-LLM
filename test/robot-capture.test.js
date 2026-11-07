import test from "node:test";
import assert from "node:assert/strict";
import { Game, PADDLE_Y, R } from "../src/game.js";
import { Network, rng } from "../src/network.js";
import { transition, train } from "../src/trainer.js";
import { DEFAULT_REWARDS, BREAKOUT_REWARDS, validateRewards } from "../src/rewards.js";
import { CaptureGame, CTF_ENV, CTF_SPEC, CTF_INPUT_NAMES, CTF_OUTPUT_NAMES, CTF_REWARDS, FLAG, BASE, WALLS, captureInputs, captureTransition } from "../src/ctf-game.js";

const zeros = (fields) => Object.fromEntries(fields.map(({ key }) => [key, 0]));
const quietMission = () => {
  const game = new CaptureGame(rng());
  game.state = "playing";
  for (const enemy of game.enemies) enemy.health = 0;
  return game;
};

test("every Breakout reward is configurable, including zero and reversed incentives", () => {
  const game = new Game(rng());
  game.launch();
  game.ball = { x: 450, y: PADDLE_Y - R - 1, vx: 0, vy: 4 };
  const hit = transition(game, 1, { ...zeros(BREAKOUT_REWARDS), paddleHit: 7 });
  assert.equal(hit.reward, 7);
  assert.equal(hit.terms.paddleHit, 7);
  game.ball = { x: 450, y: 560, vx: 0, vy: 4 };
  const miss = transition(game, 1, { ...zeros(BREAKOUT_REWARDS), missedBall: 11 });
  assert.equal(miss.reward, 11);
  assert.equal(miss.done, true);

  const brickGame = new Game(rng()); brickGame.launch();
  const b = brickGame.bricks[40];
  brickGame.ball = { x: b.x + 20, y: b.y + b.h + R + 1, vx: 0, vy: -4 };
  const brick = transition(brickGame, 1, { ...zeros(BREAKOUT_REWARDS), brickPoint: -2 });
  assert.equal(brick.reward, -20);
});

test("reward validation rejects nonnumeric and unknown terms without mutating defaults", () => {
  for (const value of [NaN, Infinity, "2", null, 1000001])
    assert.throws(() => validateRewards({ paddleHit: value }));
  assert.throws(() => validateRewards({ typo: 3 }));
  const custom = validateRewards({ alignment: 0 });
  assert.equal(custom.alignment, 0);
  assert.equal(custom.missedBall, -3);
  assert.equal(DEFAULT_REWARDS.alignment, 0.8);
});

test("Robot Capture has exactly 96 finite named inputs and 24 unique action outputs", () => {
  const game = new CaptureGame(rng()); game.state = "playing";
  assert.equal(new Set(CTF_INPUT_NAMES).size, 96);
  assert.equal(new Set(CTF_OUTPUT_NAMES).size, 24);
  for (let action = 0; action < 24; action++) {
    captureTransition(game, action);
    const input = captureInputs(game);
    assert.equal(input.length, 96);
    assert.ok(input.every((v) => Number.isFinite(v) && v >= -1 && v <= 1));
  }
});

test("variable-depth robot networks roundtrip, learn, and preserve fixed interfaces", () => {
  for (const hidden of [[3], [3, 4, 3, 2]]) {
    const network = new Network(null, rng(), [96, ...hidden, 24], CTF_SPEC);
    const input = captureInputs(new CaptureGame(rng()));
    const loaded = Network.load(JSON.parse(JSON.stringify(network.serialize())), CTF_SPEC);
    assert.deepEqual([...loaded.predict(input)], [...network.predict(input)]);
    const before = Math.abs(loaded.predict(input)[23] - 2);
    for (let i = 0; i < 100; i++) {
      loaded.clearGradients(); loaded.accumulate(input, 23, 2); loaded.update(1);
    }
    assert.ok(Math.abs(loaded.predict(input)[23] - 2) < before);
    assert.throws(() => Network.load(network.serialize()));
  }
  const maximum = new Network(null, rng(), [96, 128, 128, 128, 128, 24], CTF_SPEC);
  assert.equal(maximum.weights.flat(2).length, 65048);
  assert.deepEqual(Network.load(maximum.serialize(), CTF_SPEC).shape, maximum.shape);
  for (const shape of [[96, 0, 24], [96, 129, 24], [96, 2, 2, 2, 2, 2, 24]])
    assert.throws(() => Network.load({ ...maximum.serialize(), shape }, CTF_SPEC));
});

test("flag pickup switches the objective; delivering wins with editable rewards", () => {
  const game = quietMission();
  Object.assign(game.player, FLAG);
  const picked = captureTransition(game, null, { ...zeros(CTF_REWARDS), flagPickup: 17 });
  assert.equal(picked.reward, 17);
  assert.equal(picked.done, false);
  assert.equal(game.carrying, true);
  assert.equal(game.goal(), BASE);
  assert.equal(game.score, 25);
  Object.assign(game.player, BASE);
  const delivered = captureTransition(game, null, { ...zeros(CTF_REWARDS), delivery: 51 });
  assert.equal(delivered.reward, 51);
  assert.equal(delivered.done, true);
  assert.equal(game.state, "won");
  assert.equal(game.score, 125);
});

test("dash consumes energy and cannot cross a wall", () => {
  const game = quietMission(), wall = WALLS[0];
  Object.assign(game.player, { x: wall.x - 13, y: wall.y + 50, energy: 100 });
  captureTransition(game, 18); // Dash east.
  assert.ok(game.player.x <= wall.x - 12);
  assert.ok(game.player.energy < 81);
  assert.ok(game.player.dashCooldown > 0);
});

test("combat applies editable hit, defeat, damage, and terminal rewards", () => {
  const game = quietMission();
  const enemy = game.enemies[0];
  Object.assign(enemy, { x: 140, y: 570, health: 20, cooldown: 9999 });
  game.bullets = [{ x: 134, y: 570, vx: 8, vy: 0, enemy: false, life: 10 }];
  const result = captureTransition(game, null, { ...zeros(CTF_REWARDS), enemyHit: 4, enemyDefeat: 9 });
  assert.equal(result.reward, 13);
  assert.equal(game.hits, 1);
  assert.equal(game.kills, 1);
  game.player.health = 10;
  game.bullets = [{ x: game.player.x, y: game.player.y, vx: 0, vy: 0, enemy: true, life: 10 }];
  const lost = captureTransition(game, null, { ...zeros(CTF_REWARDS), damage: -2, defeat: -30 });
  assert.equal(lost.reward, -50);
  assert.equal(lost.done, true);
  assert.equal(game.state, "lost");
});

test("supply collection requires a deficit and awards the configured coefficient", () => {
  const game = quietMission(), supply = game.supplies[0];
  Object.assign(game.player, { x: supply.x, y: supply.y });
  captureTransition(game, null);
  assert.equal(game.pickups, 0);
  game.player.health = 50;
  const result = captureTransition(game, null, { ...zeros(CTF_REWARDS), supply: 6 });
  assert.equal(result.reward, 6);
  assert.equal(game.player.health, 90);
  assert.equal(game.pickups, 1);
  assert.ok(supply.cooldown > 0);
});

test("zero rewards disable every incentive while physics and score keep working", () => {
  const game = quietMission();
  Object.assign(game.player, FLAG);
  const result = captureTransition(game, null, zeros(CTF_REWARDS));
  assert.equal(result.reward, 0);
  assert.equal(game.carrying, true);
  assert.equal(game.score, 25);
});

test("a real training run uses chosen rewards, reports contributions, and exports its config", () => {
  const initial = new Network(null, rng(), [96, 2, 24], CTF_SPEC).serialize();
  initial.untrained = true;
  const settings = zeros(CTF_REWARDS);
  const progress = [];
  const result = train({ episodes: 1, seed: 17, environment: CTF_ENV,
    initialModel: initial, rewardConfig: settings, checkpointSelection: "latest",
    onProgress: (p) => progress.push(p),
  });
  assert.equal(progress[0].epsilon, 1);
  assert.equal(progress.at(-1).episode, 1);
  assert.ok(progress.at(-1).frames > 0);
  assert.equal(progress.at(-1).reward, 0);
  assert.ok(Object.values(progress.at(-1).rewardTerms).every((v) => v === 0));
  assert.deepEqual(result.rewardConfig, settings);
  assert.equal(result.checkpointSelection, "latest");
  assert.equal(result.checkpointEpisode, 1);
  assert.equal(result.discountFactor, 0.99);
  assert.equal(result.actionFrames, 4);
  assert.ok(Network.load(result, CTF_SPEC));
});

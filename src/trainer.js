import { Network, rng, ACTIONS } from "./network.js";
import { Game, inputs, W } from "./game.js";
export const ACTION_FRAMES = 4,
  GAMMA = 0.99;
const CAPACITY = 30000,
  BATCH = 32;
export function potential(game) {
  return -Math.abs(game.ball.x - game.paddle) / 100;
}
// A transition advances the actual physics, with no rendering or wall-clock wait.
export function transition(game, action) {
  const oldScore = game.score,
    oldHits = game.hits,
    oldLives = game.lives,
    phi = potential(game);
  let frames = 0;
  for (; frames < ACTION_FRAMES && game.state === "playing"; frames++)
    game.step(1, ACTIONS[action]);
  const done = game.lives < oldLives || game.state === "won";
  const reward =
    (game.score - oldScore) * 0.025 +
    (game.hits - oldHits) * 2 -
    (game.lives < oldLives ? 3 : 0) +
    (game.state === "won" ? 10 : 0) +
    0.8 * ((done ? 0 : GAMMA * potential(game)) - phi) -
    0.025 * (game.ball.vy > 0 ? Math.abs(game.ball.x - game.paddle) / W : 0);
  return { reward, done, frames };
}
function newEpisode(random) {
  const g = new Game(random);
  g.fast = true;
  g.paddle = 80 + random() * (W - 160);
  g.serve();
  g.launch();
  return g;
}
// Evaluation uses fixed held-out seeds, no exploration and no parameter updates.
export function evaluate(network, games = 5, maxFrames = 3600) {
  let score = 0,
    hits = 0;
  for (let i = 0; i < games; i++) {
    const g = newEpisode(rng(9000 + i));
    for (let frames = 0; frames < maxFrames; frames += ACTION_FRAMES) {
      const t = transition(g, network.action(inputs(g.ball, g.paddle)));
      if (t.done) break;
    }
    score += g.score;
    hits += g.hits;
  }
  return { score: score / games, hits: hits / games };
}
export function train({
  episodes = 2500,
  onProgress = () => {},
  onCheckpoint = () => {},
  initialModel = null,
  resetWeights = false,
  seed = 2026,
} = {}) {
  if (!Number.isInteger(episodes) || episodes < 1 || episodes > 100000)
    throw new Error("Choose between 1 and 100,000 episodes.");
  const random = rng(seed),
    network =
      initialModel && !resetWeights
        ? Network.load(initialModel)
        : new Network(null, random, initialModel?.shape),
    target = new Network(null, random, network.shape);
  target.copyFrom(network);
  const replay = new Array(CAPACITY);
  let count = 0,
    cursor = 0,
    steps = 0,
    frames = 0,
    best = -Infinity,
    checkpoint,
    loss = 0,
    updates = 0;
  const started = performance.now();
  let evaluation = evaluate(network),
    recent = [],
    epsilon = initialModel && !resetWeights ? 0.2 : 1;
  const report = (episode) =>
    onProgress({
      episode,
      episodes,
      score: evaluation.score,
      hits: evaluation.hits,
      frames,
      epsilon,
      loss: updates ? loss / updates : 0,
      framesPerSecond: Math.round(
        frames / Math.max(0.001, (performance.now() - started) / 1000),
      ),
      reward: recent.reduce((a, b) => a + b, 0) / Math.max(1, recent.length),
    });
  checkpoint = structuredClone(network.serialize());
  best = evaluation.score;
  report(0);
  const decay = 25000;
  const checkpointData = (episode) => ({
    ...checkpoint,
    trainedAt: new Date().toISOString(),
    episodes: episode,
    frames,
    score: best,
    method: "double-dqn",
    rewards: "bricks + paddle hits - missed balls + alignment shaping",
    actionFrames: ACTION_FRAMES,
  });
  for (let episode = 1; episode <= episodes; episode++) {
    const game = newEpisode(random);
    let episodeReward = 0;
    for (let t = 0; t < 900; t++) {
      const state = inputs(game.ball, game.paddle);
      epsilon =
        0.03 +
        (initialModel && !resetWeights ? 0.17 : 0.97) *
          Math.exp(-steps / decay);
      const action =
        random() < epsilon ? Math.floor(random() * 3) : network.action(state);
      const result = transition(game, action),
        next = inputs(game.ball, game.paddle);
      // A time-limit truncation still bootstraps from the resulting state.
      replay[cursor] = {
        state,
        action,
        reward: result.reward,
        next,
        done: result.done,
      };
      cursor = (cursor + 1) % CAPACITY;
      count = Math.min(count + 1, CAPACITY);
      steps++;
      frames += result.frames;
      episodeReward += result.reward;
      if (count >= 512 && steps % 4 === 0) {
        network.clearGradients();
        for (let j = 0; j < BATCH; j++) {
          const s = replay[Math.floor(random() * count)];
          const nextAction = network.action(s.next);
          const value = s.done
            ? s.reward
            : s.reward + GAMMA * target.predict(s.next)[nextAction];
          loss += network.accumulate(s.state, s.action, value);
        }
        network.update(BATCH);
        updates += BATCH;
        if (network.updates % 400 === 0) target.copyFrom(network);
      }
      if (result.done) break;
    }
    recent.push(episodeReward);
    if (recent.length > 100) recent.shift();
    if (episode % 100 === 0 || episode === episodes) {
      evaluation = evaluate(network);
      if (
        evaluation.score > best ||
        (evaluation.score === best &&
          evaluation.hits >= (checkpoint.evalHits || 0))
      ) {
        best = evaluation.score;
        checkpoint = {
          ...structuredClone(network.serialize()),
          episode,
          evalHits: evaluation.hits,
        };
        onCheckpoint(checkpointData(episode));
      }
    }
    if (episode % 10 === 0 || episode === episodes) report(episode);
  }
  return checkpointData(episodes);
}

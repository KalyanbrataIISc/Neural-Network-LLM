import { Network, rng, ACTIONS, BREAKOUT_SPEC } from "./network.js";
import { Game, inputs, W } from "./game.js";
import { DEFAULT_REWARDS, validateRewards } from "./rewards.js";
export const ACTION_FRAMES = 4,
  GAMMA = 0.99;
const CAPACITY = 30000,
  BATCH = 32;
export function potential(game) {
  return -Math.abs(game.ball.x - game.paddle) / 100;
}
// A transition advances the actual physics, with no rendering or wall-clock wait.
export function transition(game, action, rewards = DEFAULT_REWARDS) {
  const oldScore = game.score,
    oldHits = game.hits,
    oldLives = game.lives,
    phi = potential(game);
  let frames = 0;
  for (; frames < ACTION_FRAMES && game.state === "playing"; frames++)
    game.step(1, ACTIONS[action]);
  const done = game.lives < oldLives || game.state === "won";
  const terms = {
    brickPoint: (game.score - oldScore) * rewards.brickPoint,
    paddleHit: (game.hits - oldHits) * rewards.paddleHit,
    missedBall: game.lives < oldLives ? rewards.missedBall : 0,
    clearedBoard: game.state === "won" ? rewards.clearedBoard : 0,
    alignment: rewards.alignment * ((done ? 0 : GAMMA * potential(game)) - phi),
    descendingDistance: rewards.descendingDistance * (game.ball.vy > 0 ? Math.abs(game.ball.x - game.paddle) / W : 0),
  };
  const reward = Object.values(terms).reduce((sum, value) => sum + value, 0);
  return { reward, terms, done, frames };
}
function newEpisode(random) {
  const g = new Game(random);
  g.fast = true;
  g.paddle = 80 + random() * (W - 160);
  g.serve();
  g.launch();
  return g;
}
export const BREAKOUT_ENV = {
  spec: BREAKOUT_SPEC, defaultShape: [5, 24, 24, 3],
  createEpisode: newEpisode,
  inputs: (game) => inputs(game.ball, game.paddle),
  transition, validateRewards,
  score: (game) => game.score,
  hits: (game) => game.hits,
};
// Evaluation uses fixed held-out seeds, no exploration and no parameter updates.
export function evaluate(network, games = 5, maxFrames = 3600, environment = BREAKOUT_ENV) {
  let score = 0,
    hits = 0;
  for (let i = 0; i < games; i++) {
    const g = environment.createEpisode(rng(9000 + i));
    for (let frames = 0; frames < maxFrames; frames += ACTION_FRAMES) {
      const t = environment.transition(g, network.action(environment.inputs(g)));
      if (t.done) break;
    }
    score += environment.score(g);
    hits += environment.hits(g);
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
  rewardConfig,
  checkpointSelection = "score",
  environment = BREAKOUT_ENV,
} = {}) {
  if (!Number.isInteger(episodes) || episodes < 1 || episodes > 100000)
    throw new Error("Choose between 1 and 100,000 episodes.");
  if (!["score", "latest"].includes(checkpointSelection))
    throw new Error("Choose score or latest checkpoint selection.");
  const rewards = environment.validateRewards(rewardConfig ?? initialModel?.rewardConfig ?? {});
  // Validate resumed models even when the user requests randomized weights.
  if (initialModel) Network.load(initialModel, environment.spec);
  const resuming = initialModel && !resetWeights && !initialModel.untrained;
  const random = rng(seed),
    network =
      initialModel && !resetWeights
        ? Network.load(initialModel, environment.spec)
        : new Network(null, random, initialModel?.shape || environment.defaultShape, environment.spec),
    target = new Network(null, random, network.shape, environment.spec);
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
  let evaluation = evaluate(network, 5, 3600, environment),
    recent = [],
    epsilon = resuming ? 0.2 : 1;
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
      reward: recent.reduce((sum, item) => sum + item.reward, 0) / Math.max(1, recent.length),
      rewardTerms: Object.fromEntries(Object.keys(rewards).map((key) => [key,
        recent.reduce((sum, item) => sum + item.terms[key], 0) / Math.max(1, recent.length),
      ])),
    });
  checkpoint = { ...structuredClone(network.serialize()), evalHits: evaluation.hits };
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
    rewardConfig: { ...rewards },
    checkpointEpisode: checkpoint.episode || 0,
    checkpointSelection,
    discountFactor: GAMMA,
    trainingSeed: seed,
    evaluationSeeds: [9000, 9001, 9002, 9003, 9004],
    actionFrames: ACTION_FRAMES,
  });
  for (let episode = 1; episode <= episodes; episode++) {
    const game = environment.createEpisode(random);
    let episodeReward = 0;
    const episodeTerms = Object.fromEntries(Object.keys(rewards).map((key) => [key, 0]));
    for (let t = 0; t < 900; t++) {
      const state = environment.inputs(game);
      epsilon =
        0.03 +
        (resuming ? 0.17 : 0.97) *
          Math.exp(-steps / decay);
      const action =
        random() < epsilon ? Math.floor(random() * network.spec.outputs.length) : network.action(state);
      const result = environment.transition(game, action, rewards),
        next = environment.inputs(game);
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
      for (const key of Object.keys(rewards)) episodeTerms[key] += result.terms[key];
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
    recent.push({ reward: episodeReward, terms: episodeTerms });
    if (recent.length > 100) recent.shift();
    if (episode % 100 === 0 || episode === episodes) {
      evaluation = evaluate(network, 5, 3600, environment);
      if (
        checkpointSelection === "latest" ||
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

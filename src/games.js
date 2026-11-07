import { Game, W, H, PADDLE_Y, PADDLE_W, R, clamp, inputs } from "./game.js";
export const OBSERVATION_NAMES = [
  "target_x",
  "target_y",
  "target_vx",
  "target_vy",
  "agent_x",
];
export const OBSERVATION_SCHEMA = "paddle-tracking-v1";
export const GAMES = {
  breakout: { name: "Breakout" },
  catch: { name: "Catch" },
};
// Both tasks expose [target x, target y, target vx, target vy, agent x].
// The historical JSON names ball_x/... are retained for existing checkpoints.
export const observe = (game) => inputs(game.ball, game.paddle);
export class CatchGame extends Game {
  reset() {
    super.reset();
    this.id = "catch";
    this.bricks = [];
  }
  serve() {
    this.ball = {
      x: 30 + this.random() * (W - 60),
      y: 25,
      vx: (this.random() * 2 - 1) * 2.5,
      vy: 3.5 + this.random() * 1.5,
    };
    this.trail = [];
  }
  step(dt, movement = 0, target = null) {
    if (!["playing", "ready"].includes(this.state)) return;
    this.paddle = clamp(
      this.paddle +
        (target === null
          ? movement * 8 * dt
          : clamp(target - this.paddle, -8 * dt, 8 * dt)),
      PADDLE_W / 2,
      W - PADDLE_W / 2,
    );
    if (this.state === "ready") return;
    const b = this.ball,
      oldY = b.y;
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    if (b.x < R) {
      b.x = R;
      b.vx = Math.abs(b.vx);
    }
    if (b.x > W - R) {
      b.x = W - R;
      b.vx = -Math.abs(b.vx);
    }
    if (
      b.vy > 0 &&
      oldY + R <= PADDLE_Y &&
      b.y + R >= PADDLE_Y &&
      Math.abs(b.x - this.paddle) < PADDLE_W / 2 + R
    ) {
      this.hits++;
      this.score += 10;
      this.serve();
      return;
    }
    if (!this.fast) {
      this.trail.push({ x: b.x, y: b.y });
      if (this.trail.length > 12) this.trail.shift();
    }
    if (b.y > H + R) {
      this.lives--;
      this.state = this.lives ? "ready" : "lost";
      this.serve();
    }
  }
}
export function createGame(id = "breakout", random = Math.random) {
  if (!GAMES[id]) throw new Error("Unknown game: " + id);
  const game = id === "catch" ? new CatchGame(random) : new Game(random);
  game.id = id;
  return game;
}
export function schedule(plan, order = "interleaved") {
  if (
    !Array.isArray(plan) ||
    !plan.length ||
    !["interleaved", "sequential"].includes(order)
  )
    throw new Error("Choose a valid training plan and order.");
  const seen = new Set();
  let total = 0;
  for (const row of plan) {
    if (
      !GAMES[row.game] ||
      seen.has(row.game) ||
      !Number.isInteger(row.trials) ||
      row.trials < 1
    )
      throw new Error(
        "Each game needs a unique entry with a positive whole number of trials.",
      );
    seen.add(row.game);
    total += row.trials;
  }
  if (total > 100000) throw new Error("Maximum 100,000 total trials.");
  if (order === "sequential")
    return plan.flatMap((row) => Array(row.trials).fill(row.game));
  const result = [];
  for (let i = 0; i < Math.max(...plan.map((r) => r.trials)); i++)
    for (const row of plan) if (i < row.trials) result.push(row.game);
  return result;
}

export const REWARD_LIMIT = 1000000;
export const BREAKOUT_REWARDS = [
  { key: "brickPoint", label: "Per game point", value: 0.025,
    description: "Multiply game points earned by this value. Breakout bricks give 10–50 points (+0.25 to +1.25 by default); a Catch return gives 10 points (+0.25)." },
  { key: "paddleHit", label: "Paddle return", value: 2,
    description: "Added for each ball returned or caught by the paddle, in addition to the game-point term." },
  { key: "missedBall", label: "Missed ball", value: -3,
    description: "Added when a life is lost. The training episode ends on the first miss." },
  { key: "clearedBoard", label: "Clear the board", value: 10,
    description: "Added when the last brick is destroyed; the episode ends." },
  { key: "alignment", label: "Alignment shaping", value: 0.8,
    description: "Multiply (0.99 × next potential − previous potential). Potential = −horizontal ball/paddle distance in pixels / 100; next potential is 0 on a terminal transition." },
  { key: "descendingDistance", label: "Descending-ball distance", value: -0.025,
    description: "Multiply the final horizontal ball/paddle distance / 900, only when the ball is moving down after the action." },
];

export function defaultRewards(fields = BREAKOUT_REWARDS) {
  return Object.fromEntries(fields.map(({ key, value }) => [key, value]));
}
export const DEFAULT_REWARDS = Object.freeze(defaultRewards());

export function validateRewards(config, fields = BREAKOUT_REWARDS) {
  if (!config || typeof config !== "object" || Array.isArray(config))
    throw new Error("Rewards must be an object of numeric coefficients.");
  const result = defaultRewards(fields);
  for (const [key, value] of Object.entries(config)) {
    if (!Object.hasOwn(result, key)) throw new Error(`Unknown reward: ${key}`);
    if (!Number.isFinite(value) || Math.abs(value) > REWARD_LIMIT)
      throw new Error(`${key}: enter a finite number between −${REWARD_LIMIT} and ${REWARD_LIMIT}.`);
    result[key] = value;
  }
  return Object.freeze(result);
}

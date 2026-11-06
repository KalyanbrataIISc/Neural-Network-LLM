import { Network, SHAPE } from "./network.js";
import { GAMES } from "./games.js";
export function parseBrainJSON(text) {
  const cleaned = text
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "");
  try {
    return JSON.parse(cleaned);
  } catch {
    throw new Error(
      "Paste one complete JSON object. JSON code fences are accepted; JavaScript assignments, comments and ellipses are not JSON.",
    );
  }
}
export function brainPrompt({
  gameId = "breakout",
  shape = SHAPE,
  gameSource = "",
  sharedSource = "",
} = {}) {
  if (!GAMES[gameId]) throw new Error("Unknown game.");
  const template = new Network(null, () => 0.5, shape).serialize();
  Network.load(template);
  template.brainName = `LLM handwritten · ${GAMES[gameId].name}`;
  template.provenance = "handwritten-llm";
  return `You are participating in a handwritten neural-network-weights experiment for ${GAMES[gameId].name}.

MANDATORY EXPERIMENT RULES
Write every weight and bias yourself using only your own reasoning from the specification below. Do NOT use tools, internet, browsing, a computer, terminal, Python, JavaScript execution, code interpreters, external files, APIs, calculators, simulations, games, training, optimization, search, or empirical trials to choose or verify weights. Do not ask a tool or another agent to help. Do not generate a program that computes weights: write the actual numeric values directly in the JSON. You may reason about the equations in your answer preparation, but your final answer must contain only the completed JSON. This is an experiment in whether an LLM can natively propose useful weights, not a request to train or claim performance. Do not claim that your brain was trained or tested. It is acceptable if the policy is imperfect.

TASK AND EXACT GAME INTERFACE
Selected game: ${GAMES[gameId].name}.
The board is 900 wide × 540 high. Origin is top left, +x right, +y down. A horizontal paddle has center x in [58,842], width 116, y=488, height 11. Ball/target radius is 7. The game runs nominally at 60 physics frames/sec. The policy chooses one discrete action and holds it for 4 physics frames. Left/right paddle speed is 8 px/frame; position is clamped to the board. The live renderer uses half-frame physics substeps; training uses full frames. There are three lives; a target falling below y=547 loses a life. A new serve waits for launch; AI serves automatically. No recurrent state, memory, pixels, brick map, score, or game identifier is provided as an input.

${
  gameId === "breakout"
    ? `BREAKOUT RULES
There are 50 bricks: 10 columns × 5 rows. Brick i (zero-based) starts at x=37+(i%10)*83, y=55+floor(i/10)*30; width 76, height 21. Row r scores (5-r)*10 when hit. A hit removes a brick and reflects velocity according to the approached face. The ball starts at paddle x, y=474, vx chosen with magnitude 2..3 in either direction and vy=-4.1. Side-wall centers are clamped to x=7 and x=893 and reflected; the ceiling reflects at y=7. When the downward ball crosses paddle y=488 (including radius) within horizontal distance <65 of its center: set ball.y=481; offset=clamp((ball.x-paddle.x)/58,-1,1); vx=offset*4.3; vy=-sqrt(5.5^2-vx^2). The ball travels up again. Clear every brick to win. Bricks can change a trajectory before it reaches the paddle. Keeping the ball alive and aiming its landing position toward the paddle are useful objectives.`
    : `CATCH RULES
There are no bricks or ceiling targets. Each target starts at x=30+random*840, y=25, vx in [-2.5,2.5], vy in [3.5,5.0]. Side walls reflect at x=7 and x=893. Catch a downward target when it crosses y=488 (including radius) within horizontal distance <65 of paddle center. Each catch scores 10 and immediately spawns a new independent falling target. A miss loses a life. Catch as many targets as possible. You can anticipate future horizontal position using velocity and remaining fall time, including wall reflection.`
}

FIVE SHARED OBSERVATIONS, IN THIS EXACT ORDER
The JSON uses legacy key names so it is compatible with existing brains; these represent general target/agent features:
1. ball_x = 2*target.x/900 - 1 (target x, usually [-1,1])
2. ball_y = 2*target.y/540 - 1 (target y, usually [-1,1])
3. velocity_x = target.vx/6
4. velocity_y = target.vy/6 (positive means falling)
5. paddle_x = 2*paddle.centerX/900 - 1 (agent x)
These values are simultaneous floating-point observations. They are NOT raw pixel coordinates. Hidden activations are recalculated for every decision.

NETWORK CALCULATION
Architecture for this response: ${shape.join(" → ")}. Inputs and outputs are fixed; hidden widths are shown in the complete template below. Hidden neuron output = tanh(bias + sum(previous_activation * named_weight)). The final actions layer is LINEAR: Q = bias + sum(previous_activation * named_weight), with no tanh or softmax. Choose the action with maximum Q. Outputs in order: move_left (−1), stay (0), move_right (+1). If values tie exactly, the earliest action wins. Q values need only rank actions correctly; they are not probabilities and need not sum to 1. A useful policy can compare predicted target position with paddle position. The JSON tag algorithm="double-dqn" identifies the compatible Q-network format; it does not mean your handwritten weights were trained.

FILE CONTRACT
Return ONE complete JSON object, no prose, Markdown, code execution, JavaScript assignment, comments, placeholders, ellipses, NaN or Infinity. Keep version=2, observationSchema="paddle-tracking-v1", algorithm="double-dqn", shape, input order, output order, all layer names, neuron names, activation names, and incoming connection names exactly as in the template. Replace weights and biases with YOUR OWN explicit finite numeric values (absolute value <=1,000,000). Zero is allowed when deliberate. Every listed neuron must have every listed incoming weight and one bias. You can change brainName to identify the LLM that authored it, e.g. "Claude handwritten Breakout". Do not omit any neuron or connection. Output actual numeric JSON, not code to generate JSON. After manual import, a human may fine-tune your proposed brain using reinforcement learning, but that must not happen during your response.

COMPLETE VALID STRUCTURE TO FILL (zeros are a schema example, not a recommended policy):
${JSON.stringify(template, null, 2)}

EXACT JAVASCRIPT GAME SOURCE FOR REFERENCE ONLY — DO NOT EXECUTE
This includes the selected task and the common physics. Read it as text to understand collisions and observations. Imports are reference text, not permission to run anything.
--- shared game.js ---
${sharedSource}
--- task adapters games.js ---
${gameSource}

FINAL REMINDER: reason and handwrite all numbers natively, without tools, browsing, running code, simulating, training, or checking performance. Return only your complete JSON brain.`;
}

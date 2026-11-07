# Your editable neural network

The actual saved network is **models/pretrained.json**. This is a normal text file, not a binary tensor. You can open it in any editor, change numbers, and use **Reload project weights** in the game. You can also edit the same structure in the game's editor. Export your experiments to keep separate versions.

## Inputs and actions

| Name         | Value received by the network |
| ------------ | ----------------------------- |
| `ball_x`     | `2 * ball.x / 900 - 1`        |
| `ball_y`     | `2 * ball.y / 540 - 1`        |
| `velocity_x` | `ball.vx / 6`                 |
| `velocity_y` | `ball.vy / 6`                 |
| `paddle_x`   | `2 * paddle.x / 900 - 1`      |

The ball origin is at the top left. Positive x velocity goes right; positive y velocity goes down.

The final layer has `move_left`, `stay`, and `move_right`. These are **Q values**, not probabilities. The largest one selects the action. Left moves the paddle at −8 pixels/frame; right moves it at +8; stay moves it at 0. The action is recomputed every four physics frames. The network itself provides the action; no hand-coded interception controller replaces it.

## Reading one neuron

Every neuron has a name, bias, and individually named incoming weights. An example first-layer neuron looks like this (illustrative values):

```json
{
  "name": "hidden_1_01",
  "bias": 0.02,
  "weights": {
    "ball_x": 0.15,
    "ball_y": -0.03,
    "velocity_x": 0.42,
    "velocity_y": -0.18,
    "paddle_x": -0.27
  }
}
```

Its activation is:

```text
sum = 0.02
    + ball_x     *  0.15
    + ball_y     * -0.03
    + velocity_x *  0.42
    + velocity_y * -0.18
    + paddle_x   * -0.27

hidden_1_01 = tanh(sum)
```

Second-layer weights refer to first-layer neuron names. Action-layer weights refer to second-layer neuron names. The action layer uses the sum directly, without tanh.

## An experiment with an obvious effect

1. Open **Edit neurons & weights** and select **Actions**.
2. Set every incoming weight for `move_right` to 0 and its bias to 100.
3. Apply the edits, close the editor, and click **Watch AI play**.
4. The paddle will favor moving right because that action now has a very high Q value.
5. Click **Reload project weights** to restore the checkpoint on disk, or import a saved experiment.

Each number affects the actual feed-forward computation. No retraining is required to try manual edits. Large weights can saturate hidden tanh neurons; that is part of the experiment.

## Adding neurons

The editor's **Add neuron in this layer** button keeps the structure valid automatically. It adds a neuron with small random incoming weights and zero outgoing weights. Until you edit the outgoing connections or train, that new neuron contributes nothing to later layers. Removing a neuron removes its outgoing connections too.

For hand-editing JSON, keep all of these in agreement:

- `shape`: `[5, first_hidden_count, second_hidden_count, 3]`
- Each layer's `neurons` list, with names numbered continuously from `hidden_1_01` or `hidden_2_01`
- Every next-layer neuron's weight map must contain one entry for every neuron in the previous layer
- `hidden_1` and `hidden_2` use `tanh`; `actions` uses `linear`
- Input and output names must stay in the documented order

The two hidden counts can each be 1–64. Values must be finite numbers with magnitude at most 1,000,000. JSON cannot contain comments, `NaN`, or `Infinity`. The top-level `version: 2` and `algorithm: "double-dqn"` identify this checkpoint format.

Breakout supports exactly two hidden layers: four layers in total when counting the fixed input and output layers, or three parameterized layers. Depth cannot be changed through the editor or JSON. The maximum architecture is `5 → 64 → 64 → 3`, with 4,739 parameters. The browser import limit is 1,500,000 bytes. The UI displays these constraints. Robot Capture has a separate 96-input/24-output interface and larger architecture limits; see [ROBOT_CAPTURE.md](ROBOT_CAPTURE.md).

The file also includes descriptive training metadata such as episodes, simulated frames, and evaluation score. These fields are not input to the neurons. Editing the network through the UI clears old evaluation metadata so it is not presented as a measurement of your edited policy.

## Training your edited architecture

Apply the edits, then click **Train neural network**. The learner starts with that actual structure and those weights. It explores actions in fast simulations and updates weights from transition rewards. Or export the model and use:

```sh
npm run train -- --episodes 25000 --resume my-weights.json --out models/experiment.json
```

The source in `src/network.js` spells out the same neuron operations and optimization in JavaScript without an ML framework.

## The complete Breakout reward

For each action, up to four physics frames are simulated. Let `phi = -abs(ball.x - paddle.x) / 100`. The six coefficients below are editable under **Reinforcement rewards · view & edit**:

```text
reward = brickPoint × pointsEarned
       + paddleHit × paddleReturns
       + missedBall × lifeLost
       + clearedBoard × boardCleared
       + alignment × (0.99 × nextPhi - previousPhi)
       + descendingDistance × (ballMovingDown ? abs(ball.x - paddle.x) / 900 : 0)
```

Defaults are `brickPoint: 0.025`, `paddleHit: 2`, `missedBall: -3`, `clearedBoard: 10`, `alignment: 0.8`, and `descendingDistance: -0.025`. Events are counted over the action; the descending feature uses the final state. On a terminal transition (first lost life or cleared board), `nextPhi` is zero. At the 3,600-frame time limit there is no additional reward and the next Q value still bootstraps. Rewards are summed without clipping. Huber loss clips the derivative of temporal-difference error to [−1, 1]; it does not clip reward.

The UI locks a validated reward snapshot during each run and saves the exact coefficients as `rewardConfig` in new checkpoints. Mean episode reward and its six mean contributions use a rolling window of up to 100 completed episodes. Evaluation remains average brick points over five fixed-seed games, independent of custom rewards; in best-points mode the checkpoint is chosen by evaluation points, then paddle returns. The alternative **Latest evaluated policy** mode saves the newest evaluated weights even when points fall, making reward experiments easier to inspect. `checkpointEpisode` identifies which evaluation checkpoint was retained, and can be zero when the run's initial weights remained best. Changing rewards requires training to affect network behavior.

# Robot Capture

Run `npm start` and open **http://localhost:3000/ctf.html**, or follow **Try Robot Capture** from Breakout. This is a separate experiment: checkpoints and browser storage are separate from Breakout.

## Mission and controls

One robot must retrieve the flag and bring it to its home base on the selected map. Objective positions, obstacles, sentries, and supplies vary across 10,000 seeded arenas. Four scripted sentries patrol and shoot. Walls stop both robots and projectiles. Health, ammunition, and energy stations refill resources and become available again after 600 physics frames. A mission ends at delivery, destruction, or 3,600 frames (60 simulated seconds).

Use WASD or arrows to move, Space to fire, and Shift to dash. Touch devices have direction, fire, and dash buttons. Space/Shift without a movement key uses the current facing direction. Fire always uses the same aiming assist for humans and the network: aim at the nearest visible living enemy; otherwise fire along the movement direction. It costs one round, with a 14-frame cooldown. Dash costs 20 energy, with a 45-frame cooldown. A dash uses collision-checked substeps. Maximum resources are 100 health, 30 rounds, and 100 energy; energy regenerates by 0.08 per frame. Carrying the flag reduces normal movement speed from 3 to 2.4 pixels/frame.

## Maps and generalization

The **Map set**, **Map ID**, and **Random map** controls browse 10,000 reproducible layouts. The selected ID is included in the URL (`/ctf.html?map=8042`) so you can return to the same geometry and spawn locations. Patrol timing can differ between missions. Changing maps restarts the mission; an active mission continues on the new map with reset resources. The default live map is **2001, a held-out test map**.

| Set | IDs | Maps | Used for |
| --- | --- | ---: | --- |
| Training | 1–1000 | 1,000 | Randomly sampled training episodes, with replacement |
| Validation | 1001–2000 | 1,000 | Checkpoint evaluation uses fixed IDs 1001–1005; other validation maps remain browsable |
| Held-out test | 2001–10000 | 8,000 | Play or watch the policy on maps excluded from training and checkpoint selection |

The split is enforced by the environment factory, independent of the currently displayed map. Watching a test map while training does not put that map into replay memory. Validation is used to select checkpoints, so it is distinct from the final held-out test pool. Models record the generator version, split boundaries, validation IDs, and sorted `trainingMapIds` visited in their run history. Resetting weights clears that history. Imported or older weights without this metadata have an unknown map history; the interface says so instead of claiming they are unseen.

The versioned `ctf-maps-v1` generator builds a connected maze, adds loops, and ensures a collision-clear short route between base and flag. Four sentries and four supply stations have clear, distinct spawn positions away from the walls. All layouts are generated on demand; the app does not download thousands of level files. Connectivity guarantees a geometric route, not survival against the sentries. The network is given its normal sensors and objective offsets, never a pathfinding solution. Map 0 remains an internal legacy physics fixture and is excluded from the browser and every training/evaluation split.

## Live neural network map

The robot's brain appears beside the arena on a wide screen and below it on a narrow screen. It shows **all 96 inputs, every hidden neuron, and all 24 action outputs**. Live values come from the actual policy. Replay freezes one inference calculation while signals travel through the layers; worker snapshots show the training network and its replay sample. The stage label distinguishes these two sources.

Select a neuron to show all its connections and inspect its sum, bias, and activation. Hover to see names and values; right-click a hidden/output neuron or connection to edit its real bias or weight. Applying a graph edit stops active training, installs the edited controller, saves it locally, and clears stale evaluation/map-history metadata. Drag from one neuron to the next layer to edit a connection, Shift-drag to arrange, or use the neuron palette to add a hidden neuron (up to 128 per layer). New outgoing connections start at zero. Architecture/JSON editing is also available from the panel.

The overview samples connections for readability and shows compact labels on dense layers; every neuron is still present, and selecting a neuron exposes all its connections. On narrow screens the diagram scrolls horizontally. Replay speed and **Step layer** control the visualization independently of gameplay. **Train robot**, **Watch AI play**, and **Training settings** are directly available above the diagram.

## Network interfaces and limits

The default network is **96 → 48 → 32 → 24**, with **7,016 parameters**. Robot Capture supports **1–4 hidden layers, each with 1–128 neurons**, plus a fixed input and output layer (3–6 layers in total). The maximum is **65,048 parameters**. These are implementation limits, not universal neural-network limits. Hidden layers use tanh; outputs are linear Q values. The highest output selects an action for four frames.

**Edit architecture & weights** lets you rebuild with comma-separated hidden sizes. Rebuilding resets all weights; the UI explicitly says so. For specific weight edits, change the complete named JSON and apply it. Applying validates it, stops training, installs the actual weights, and removes stale evaluation metadata. Every weight/bias must be a finite number between −1,000,000 and +1,000,000. Imported files may be at most 8,000,000 bytes.

The 96 inputs are defined, in exact order, by `CTF_INPUT_NAMES` and `captureInputs` in `src/ctf-game.js`:

| Group | Count | Contents |
| --- | ---: | --- |
| Robot and objectives | 16 | Position, velocity, health, energy, ammo, fire/dash cooldowns, flag possession, offsets to flag/home, facing |
| Directional ranges | 32 | Eight compass sectors × wall, visible enemy, visible supply, and visible hostile projectile range |
| Enemy slots | 32 | Four fixed sentry slots × relative x/y, velocity x/y, health, range, visibility, alive |
| Known supply stations | 16 | Four stations × relative x/y, type, availability |

Coordinates and relative offsets scale by map width (960) and height (640). Health and energy scale by 100, ammo by 30, robot velocity by 18, enemy velocity by 1.2, and enemy health by 60. Cooldowns scale by 14/45. Distances scale by 1,000, with 1 meaning no directional detection. Wall rays intersect rectangles exactly; object detection uses eight 45° sectors. Enemy features are zero when occluded except for alive state. Supply locations and availability are known. Supply type encodes health as 0, ammo as 0.5, energy as 1. All observations are clamped to [−1, 1]. The page exposes every live input and Q value.

The **24 outputs** are eight directions (N, NE, E, SE, S, SW, W, NW) for each ability (move, fire, dash). Indices 0–7 move, 8–15 fire while moving, and 16–23 dash while moving. All AI actions move; there is no idle action. Human control additionally permits idle. No pathfinding or example actions are provided to the learner.

## Editable rewards

Every coefficient is editable on the page. Zero disables a term; negative values reverse its incentive. Defaults:

| Event or feature | Coefficient | Calculation |
| --- | ---: | --- |
| Flag pickup | +8 | Once when the flag is acquired |
| Delivery | +30 | Once when carrying the flag into home base |
| Enemy projectile hit | +1 | Per hit, which deals 20 damage |
| Enemy defeated | +3 | Per defeat, additional to the hit reward |
| Damage received | −0.15 | Per health point actually lost |
| Useful supply collected | +0.5 | Per pack, only when its resource is below maximum |
| Objective progress | +2 | × (distance before − distance after) / 100, using the objective selected before the action |
| Robot destroyed | −10 | At zero health |
| Time cost | −0.002 | Once per action |

The sum of these terms is the entire, unclipped reward for the transition. There is no separate timeout reward. Delivery/destruction stop Q-value bootstrapping; a time-limit truncation still bootstraps. Edits change the next training run, not existing weights. During live play the page previews total and per-term rewards using the current editor coefficients; that preview performs no learning. Training reports the mean total and each contribution over the most recent 100 episodes (or all completed episodes if fewer).

Settings save separately from weights in browser storage. The trainer snapshots and validates them at run start; the UI locks them during training. Exported checkpoints record `rewardConfig`, discount factor, training/evaluation seeds, and `checkpointEpisode` (0 if the best retained weights were the run's initial policy). The saved run's reward configuration is displayed separately from next-run settings. Importing a model does not overwrite your next-run settings. Older checkpoints without recorded settings say so.

## Training and comparison

This experiment starts with random weights; there is no bundled trained robot. It uses the same Double DQN implementation as Breakout: discount 0.99, a 30,000-transition replay buffer, batches of 32, Huber loss, Adam at learning rate 0.0006, updates every four action decisions after 512 transitions, and target-network synchronization every 400 optimizer updates. Fresh/randomized policies explore from 100%; resumed policies from 20%; exploration decays toward 3% with a 25,000-decision time constant. Each episode permits 900 action decisions (3,600 frames).

Evaluation uses map IDs 1001–1005 with simulation seeds 9000–9004, no exploration, and no weight updates. Training draws maps only from IDs 1–1000; the 8,000 held-out test maps never select checkpoints. **Mission points are separate from reward:** +5 per enemy hit, +15 per defeat, +25 for flag pickup, +100 for delivery. **Policy to save** defaults to best mean mission points, then hit count, independent of your custom reward. You can instead choose **Latest evaluated policy** to inspect reward-driven behavior even if mission points fall. Both modes save at evaluation boundaries (every 100 episodes and at completion). Best-points mode can retain the initial policy; latest mode retains the final evaluated weights. The reward editor does not change mission points or the opponent rules. A high score can reflect combat without flag delivery; watch the live mission to judge the policy. This larger task can require substantial training and reward/architecture experimentation.

For an A/B experiment, export the first result, change one reward, select **Reset weights**, and use the same architecture and episode budget. Training always starts with seed 2026 in the browser; reset runs of the same setup are repeatable. Different reward-driven actions can still lead to different trajectories and later random draws.

Terminal training is also supported:

```sh
npm run train -- --game ctf --episodes 500 --out models/robot-capture.json
npm run train -- --game ctf --episodes 500 --resume models/robot-capture.json --rewards my-rewards.json --out models/robot-experiment.json
```

`my-rewards.json` is an object of coefficients, for example `{"flagPickup": 12, "delivery": 50, "progress": 0}`. Missing fields use game defaults; unknown fields and non-finite/out-of-range values are rejected. Without `--rewards`, a resumed run uses its recorded configuration when available. Add `--save-policy latest` to save the latest evaluated policy. Import the output file into Robot Capture to watch it.

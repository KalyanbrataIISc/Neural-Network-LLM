# Neural Network and LLMs

Play Breakout with your keyboard, train a tiny network with **reinforcement learning in fast-forwarded simulations**, then watch its saved policy play live. Edit individual weights, biases, and hidden-neuron counts and immediately try the result.

## Run on your Mac

Install [Node.js](https://nodejs.org/) 22 or newer, then:

```sh
git pull origin main
npm start
```

Open **http://localhost:3000** in Chrome, Safari, or Firefox. No `npm install`, Python, ML libraries, GPU, or internet connection is needed to run it. Stop with Control-C. Use `PORT=3001 npm start` if port 3000 is occupied.

For the larger **Robot Capture** experiment, follow **Robot Capture** or open **http://localhost:3000/ctf.html**. It has **96 inputs, 24 actions, a live editable neural network map, editable rewards, and 1–4 hidden layers of 1–128 neurons each**. Its 10,000 seeded arenas are split into 1,000 training, 1,000 validation, and 8,000 held-out test maps; the live game defaults to held-out play, and training never samples that test pool. Navigate cover, fight four sentries, gather supplies, retrieve the flag, and return home. See [ROBOT_CAPTURE.md](ROBOT_CAPTURE.md) for the controls, sensors, exact reward formulas, and training behavior.

## Play and train

- **Human control:** click Start playing. Move with **← / →** or **A / D**. **Space** launches the ball or pauses/resumes. Touch users can drag on the playing field.
- **Watch AI play:** a pretrained reinforcement-learning checkpoint is included in `models/pretrained.json`. Click Watch AI play to run its actual policy in real time. AI serves automatically after a lost life. You can switch between human and AI control during a game.
- **Train neural network:** choose a simulation budget from 1,000 to 100,000 episodes. Training runs as fast as your CPU can simulate, in a Web Worker, independently of live gameplay. Millions of physics frames can run without rendering any of them. The panel shows actual simulated frames, simulation throughput, exploration probability, and periodic evaluation scores.
- When a model is already loaded, training starts from **those weights and that architecture**, including manual edits. Check **Reset weights** to start from random weights while keeping your neuron structure. With no model it starts from random weights. Adam state and replay memory start fresh for each run.
- The best evaluated checkpoint is saved during training. Stop preserves the latest checkpoint. You can watch a saved checkpoint while training continues in the background.
- Weights save in this browser's local storage. **Export weights** downloads readable JSON; **Import weights** loads it on any computer. Export before clearing browser data.

## Edit the brain

Click **Edit neurons & weights**. Every weight and bias is an editable number. Select a layer, change values, or add/remove hidden neurons. Click **Apply edits to live game**: the running controller immediately uses the new network, and the edits are saved locally. Applying manual edits stops active training so a training checkpoint cannot overwrite them.

A brain can have **1–6 hidden layers**, each containing **1–64 neurons** independently. The default pretrained model has two hidden layers. Adding a neuron creates its incoming connections and adds a new connection to every neuron in the next layer. New outgoing weights are zero, so adding alone initially preserves the policy. Change these weights to see an effect, or train the expanded model. The five game inputs and three paddle actions are fixed interfaces.

Breakout supports **1–6 hidden layers** (3–8 layers including input/output), each 1–64 neurons wide. Use the brain builder to change depth. The maximum is **21,379 parameters**. Limits are displayed beside the architecture and in the editor, with explanations when add/remove buttons are disabled. These are experiment-specific constraints.

You can also edit the **complete JSON** inside the editor, or edit `models/pretrained.json` on disk. **Reload project weights** reads that file and applies it, even if browser storage contains another model. Editing a file alone does not hot-reload it: use that button or import your edited file. All valid numeric edits execute as network parameters. Invalid shapes, names, or non-finite values produce an error while the previous controller remains usable.

See [NEURAL_NETWORK.md](NEURAL_NETWORK.md) for the exact schema and calculations.

## Inspect and change reinforcement rewards

Open **Settings & training → Reinforcement rewards · view & edit** in the training panel. Every coefficient is editable: brick points (×0.025), paddle returns (+2), missed balls (−3), board completion (+10), potential-based alignment shaping (×0.8), and descending-ball distance (×−0.025). Each field explains its exact calculation. Zero disables a term; negative values reverse incentives. Allowed values are finite numbers from −1,000,000 to +1,000,000.

Edits save in this browser and apply to the **next training run**. Retraining changes the policy; editing a reward does not immediately change saved weights. Reward inputs lock while training uses its snapshot. The panel reports mean episode reward and each term's contribution over the most recent 100 episodes, separately from evaluation game points. Exported checkpoints record the run's exact `rewardConfig` and seeds; **Saved training run rewards** shows those independently of your next-run settings. Legacy checkpoints explicitly report that exact coefficients were not recorded. Importing weights does not replace your next-run preferences.

Use **Policy to save** to choose the best evaluation points (default), or the **latest evaluated policy** to inspect behavior learned under your rewards even if game points fall. Best-points mode can retain the initial weights. Both modes checkpoint every 100 episodes and at completion; manual mode also evaluates after each Forward.

For a controlled comparison, keep the architecture/budget, select **Reset weights**, and change one reward. Browser training uses seed 2026; evaluation uses seeds 9000–9004. For terminal training, pass `--rewards my-rewards.json` with an object of coefficient overrides. A resumed terminal run uses its saved coefficients unless explicitly overridden. Add `--save-policy latest` to save the latest evaluated policy instead of the best game points.

## Train from the terminal

Same simulation, same network, no browser needed:

```sh
npm run train -- --episodes 5000
npm run train -- --episodes 25000 --resume models/pretrained.json --out models/my-experiment.json
```

The default output is `models/pretrained.json`. Progress prints every 100 episodes. Improved checkpoints are written atomically as formatted JSON, so interrupting a long run preserves the last saved policy. To use a different output file, import it in the app. A long run is computationally intensive; wall-clock duration depends on CPU speed and how long the agent survives.

## How reinforcement learning works

The default model is **5 → 24 → 24 → 3**, with **819 parameters**. The bundled checkpoint was trained from scratch for 5,000 episodes (16,113,843 simulated frames); its best checkpoint scored 854 points on the five evaluation games. Hidden neurons use tanh; output neurons produce linear Q values for left, stay, and right. The highest Q value selects an action, held for four simulated frames. The same limited paddle speed and physics drive training and live gameplay.

The learner is **Double DQN**: epsilon-greedy exploration, a 30,000-transition replay buffer, batches of 32, a separate target network, Huber loss, and Adam. It learns from actions and resulting rewards: brick points, paddle returns, missed balls, and a small alignment reward. That shaping uses the current horizontal ball/paddle distance; it never provides a target action, interception prediction, example solution, or pretrained teacher.

An episode ends on the first missed ball, a cleared board, or 3,600 simulated frames. Evaluation periodically runs five fixed-seed games with exploration disabled; the best scoring checkpoint is saved. Training rewards and evaluation score measure different things. The small network sees ball/paddle state rather than the full brick board. More training does not guarantee a perfect player; inspect the policy in the live game and experiment.

## Development

```sh
npm test
```

Tests cover collisions, terminal rewards, checkpoint validation, actual action changes after weight edits, architecture expansion, and DQN optimization. The browser flow has also been checked for keyboard play, worker training, saved weights, export/import, manual weight and architecture editing, and mobile layout.

- `src/game.js`: physics shared by simulation and visible gameplay
- `src/network.js`: neuron calculations, readable serialization, backpropagation, Adam
- `src/trainer.js`: fast simulations, rewards, replay, Double DQN, evaluation
- `src/train-worker.js`: background browser training
- `src/app.js`: live game, editor, model controls
- `scripts/train.js`: terminal training and checkpoint saves
- `server.js`: dependency-free local static server

## Visual workbench

The light workspace starts with 25% game / 75% neural network. Drag the vertical divider (or focus it and use arrow keys) to resize; your split is saved. Training, JSON import/export, and the full numeric editor live under **Settings & training**.

The network map shows every connection. Teal means a positive weight, coral means negative; color intensity shows magnitude. All lines have the same width. Click a neuron to reveal its incoming weight labels; hover any neuron or connection for exact activation, bias, or weight values. Glowing neurons and moving signals replay captured forward-pass activations. This is an explanatory replay of an instantaneous calculation, not a claim that computation physically travels at the animation's speed. In human mode the network observes the live game state as well.

Use **Playback** to slow the game and animation together, or pause. **Step layer** pauses playback and reveals input → hidden 1 → hidden 2 → action one layer at a time; completing an inference advances the game by one action interval. Training remains fast and independent of playback speed. During training the map displays current learner weight snapshots and actual replay-buffer sample inputs, separately from the best checkpoint controlling the game.

Drag a neuron from the palette into either hidden layer. Drag one neuron onto a neuron in the next layer to create/update a connection (initial weight 0.1); shift-drag to rearrange nodes visually. Connections follow neighboring layers in the feed-forward architecture. Right-click a hidden/action neuron to edit its bias, or a connection to edit its weight. Changes apply immediately; ↑/↓ nudges by 0.01. Set a connection to zero to remove its contribution. Editing stops active training to prevent it overwriting your edits; gameplay continues. Export saves the resulting numeric model, which another user can import. Visual node positions are workspace-only.

## My Brains and transfer experiments

**My Brains** is a browser-local library. Save the current network as a named brain, rename its name field, load it into the live editor, update its saved weights, export it, or delete it. Importing JSON validates the complete network and observation interface before saving a new entry. Editing a live brain does not overwrite a named copy until you click **Update loaded brain**. Browser storage is local to this browser and origin; export JSON for durable backups or sharing.

Select **Breakout** or **Catch** above the game. Switching tasks keeps the same neural network, so you can inspect transfer without retraining. Catch repeatedly drops a moving target; the paddle earns points by catching it. Both tasks expose the `paddle-tracking-v1` interface:

| Slot | Meaning across tasks                           | Historical JSON key |
| ---- | ---------------------------------------------- | ------------------- |
| 0    | Target horizontal position, normalized to −1…1 | `ball_x`            |
| 1    | Target vertical position, normalized to −1…1   | `ball_y`            |
| 2    | Target horizontal velocity / 6                 | `velocity_x`        |
| 3    | Target vertical velocity / 6                   | `velocity_y`        |
| 4    | Controlled agent horizontal position, −1…1     | `paddle_x`          |

Actions remain left / stay / right. The graph uses general target/agent labels; the readable JSON retains old key names so existing brains still load. Future tasks must provide an adapter to these five slots and three actions; incompatible interfaces are rejected rather than silently remapped.

In **Settings → Training arena**, enable a multi-game plan and assign a whole-number trial count to each task (0 excludes a task). **Interleaved** alternates tasks while respecting exact totals; **Sequential** trains all Breakout trials followed by all Catch trials. The same weights, optimizer, and replay buffer continue across task switches. Reset weights is optional. Evaluation reports each task separately; best-checkpoint selection uses the equally weighted mean of per-task scores, not the trial proportions. Checkpoints record the requested plan, actual completed counts, and history. Scores have task-specific meanings, so compare per-game results as well as the aggregate.

For terminal experiments:

```sh
npm run train -- --game catch --episodes 5000 --resume models/pretrained.json --out models/catch.json
```

Or put `[{"game":"breakout","trials":500},{"game":"catch","trials":500}]` in a plan JSON file and use `--plan path/to/plan.json --order interleaved`. These are repeated episodes, not extra neural-network inputs.

## Optional native GPU training

The default browser worker uses CPU. To train the neural-network minibatches on an Apple GPU or NVIDIA GPU, run the optional local PyTorch service. Physics and environment stepping remain on CPU; tensor forward/backward passes and Adam updates run on the selected device. The small actor used for gameplay exploration runs on CPU and refreshes from the learner periodically. Tiny networks can run faster on CPU because GPU dispatch has overhead; GPU availability is not a speed guarantee.

On an Apple Silicon Mac, in a second terminal at the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r backend/requirements.txt
npm run gpu
```

Keep `npm start` running in the first terminal. Select **Native · auto GPU** in Training arena and click **Check native device**. The service selects **CUDA** when available, otherwise **Apple MPS**, otherwise reports **CPU fallback**. Intel Macs generally fall back to CPU.

On Windows, create/activate a Python environment, install a CUDA-enabled PyTorch build using the command from the [official PyTorch installer](https://pytorch.org/get-started/locally/) for your GPU/driver, then run `python backend/service.py`. A CPU-only PyTorch build cannot use CUDA even if the machine has an NVIDIA GPU. The browser connects through the Node server; the native service binds only to `127.0.0.1:3001`. If it is absent, the UI reports setup instructions and the browser CPU trainer remains usable.

Native training supports the editable reward coefficients, reward contribution reporting, best/latest policy selection, readable brain JSON, edited hidden-layer sizes, resume/reset, mixed-task schedules, and live weight snapshots. Browser and PyTorch learners share task semantics, rewards, and checkpoint schema; random number streams, precision (PyTorch float32), and periodic actor refresh differ, so runs are not bit-for-bit identical. No simulated device labels are used: the UI reports the actual PyTorch device. Native training was verified end to end on CPU here; CUDA and MPS require testing on their respective hardware.

Optional backend tests:

```sh
python3 -m unittest discover -s backend -p 'test_*.py'
```

## Following a signal

One animation clock now controls both edge travel and layer transitions. A captured input and its weights remain stable through the complete replay; incoming live state and training snapshots wait until the next replay cycle. Pausing freezes packet position, and playback speed scales a single clock. Selecting an edge shows `source activation × weight = contribution` and places the multiplication on its traveling packet. Selecting a receiving neuron shows `Σ(activation × weight) + bias → tanh/linear → activation`. These are the numbers from that captured inference, not decorative values or the latest changing input. The live controller continues independently while its captured calculation is replayed.

## Add Brain and the handwritten LLM experiment

Open **My Brains → Add Brain**. You can:

- Build a fresh network with **1–6 hidden layers**, set each layer's neuron count, and choose small random weights or an all-zero manual template. **Generate JSON draft** lets you edit every number; **Create & edit on map** saves and opens the network for drag-and-drop edits.
- Import a JSON file, paste/write complete JSON, or copy the current brain into a separate draft. **Validate JSON** checks the shape, names, connections, numeric weights, and observation interface without playing or training it. JSON code fences from an LLM are accepted; executable JavaScript is not.
- Give the result any name, such as “Claude handwritten Breakout,” then save and load it. It becomes a regular brain that can play either task, be edited, or continue training in the arena.

**Copy LLM prompt** generates a prompt for the currently selected game and, by default, the live model's exact architecture. **View prompt** exposes the complete text, including a manual-copy fallback if clipboard permission is unavailable. The prompt includes exact observations/scaling, action timing and tie-breaking, collisions and game rules, network equations, schema constraints, a full valid zero-filled JSON template, and the current game source. It explicitly asks the LLM to choose and write every numeric weight itself and forbids tools, internet, browsing, computer/code execution, simulation, training, optimization, or programs that generate weights. It requests only a complete literal JSON object and makes no claim of trained performance.

The prompt defaults to the live model’s hidden-layer count and widths (for the bundled model: **5 → 24 → 24 → 3**). Choose **Customize layers and neuron counts** to add/remove prompt layers and set their widths independently of the scratch builder. The summary shows the requested architecture and parameter count before copying. **Reset to current model** restores its hidden widths. The supported range is 1–6 hidden layers, each 1–64 neurons. Paste the LLM response into Add Brain, validate, name it, and load it. The app validates the model's structure and values; it cannot establish whether an external LLM actually obeyed the no-tools instruction. Fine-tuning is an explicit later step in the arena.

## Main-window learning controls

The site is **Neural Network and LLMs**. Training status now stays beside the network map: trial progress, evaluation chart, per-game scores, simulation throughput, replay warmup, optimizer update count, and parameter changes. The arena configuration remains under Settings; **Configure arena** opens it without changing the live workspace.

Select **Auto** for maximum-speed simulation or **Manual** for one complete training game per **Forward** press. A simulation ends on a miss, a win, or the 3,600-frame trial limit. Manual stepping retains replay memory, target weights, and Adam state across clicks; it is not repeated fresh training jobs. **Run at maximum speed** resumes the same session. **Pause for manual stepping** pauses at an episode boundary. **Stop** ends that session; the displayed brain remains loaded. Native training supports the same controls. Manual mode evaluates each completed episode, displays its current weights (rather than an older best checkpoint), and lets that updated brain control the visible game.

The exact-zero option is **My Brains → Add Brain → All-zero brain · default 5 → 24 → 24 → 3**. It creates all 819 weights and biases as literal zero. Saving loads that exact brain. Training a completely zero tanh network cannot initially learn input-dependent hidden features: output biases can change while hidden weights remain zero because their gradients are zero. This is shown honestly in the parameter counts; no random noise is secretly injected. Choose **Reset weights** if you explicitly want to break symmetry using random initialization. Manual mode starts learning once 32 transitions are available; automatic mode warms up with 512. The replay counter explains any step that only gathers data.

**Auto-restart after win / loss** is enabled by default and saved in the browser. After a short pause, the live game resets all lives, score, ball, and bricks, then launches using the same brain. Disable it to inspect a finished game. Paused playback does not automatically advance to a new game. Training episodes remain separate from this live-game setting.

## Native GPU performance diagnostics

Automatic MPS/CUDA training now averages **512 samples every 64 environment decisions**, rather than launching a 32-sample update every four decisions. Both schedules consume eight replay samples per new transition, but the larger batches make 16 times fewer optimizer launches and change the optimization trajectory. Monitor evaluation score, not only frames/sec. Manual mode deliberately uses 32-sample updates for close inspection.

The native replay buffer is a ring list sampled directly; it no longer converts the entire 30,000-item buffer to a list for every update. A batch packs all fields into one host-to-device transfer. Actor refresh and JSON snapshots flatten parameters for a single device-to-host transfer, avoiding one synchronization per layer tensor; automatic snapshots are rate-limited. There is no per-update GPU loss readback. Target/actor refresh frequency is tied to environment decisions. These changes reduce launch, copy, and synchronization overhead; CPU game physics still sets a throughput ceiling, and an 819-parameter model can remain too small for a GPU to beat every CPU configuration.

Use **Native · force CPU (comparison)** to compare native CPU training with auto-device training on the same machine. The live panel reports the actual device, batch size, update count, and end-to-end frames/sec. The browser JavaScript CPU worker is a separate implementation, so comparing it with native MPS includes Python-versus-JavaScript simulation costs as well as GPU computation.

For a synchronized learner benchmark on your Mac:

```sh
source .venv/bin/activate
python3 backend/benchmark.py --device all --samples 16384
```

This warms up kernels first and synchronizes the device around timing. It prints real CPU/MPS/CUDA samples/sec for batch sizes 32 and 512 using the same sample budget; it excludes game physics and performs fewer Adam updates at the larger batch size. `--hidden 64,64` tests a larger network. This environment has no Apple GPU: optimized native CPU and all manual controls were tested here, while MPS speed must be measured on your Mac. Restart both Node and the Python service after pulling these changes.

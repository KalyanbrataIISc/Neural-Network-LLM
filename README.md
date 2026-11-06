# Breakout Lab

Play Breakout with your keyboard, train a tiny network with **reinforcement learning in fast-forwarded simulations**, then watch its saved policy play live. Edit individual weights, biases, and hidden-neuron counts and immediately try the result.

## Run on your Mac

Install [Node.js](https://nodejs.org/) 22 or newer, then:

```sh
git pull origin main
npm start
```

Open **http://localhost:3000** in Chrome, Safari, or Firefox. No `npm install`, Python, ML libraries, GPU, or internet connection is needed to run it. Stop with Control-C. Use `PORT=3001 npm start` if port 3000 is occupied.

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

You can also edit the **complete JSON** inside the editor, or edit `models/pretrained.json` on disk. **Reload project weights** reads that file and applies it, even if browser storage contains another model. Editing a file alone does not hot-reload it: use that button or import your edited file. All valid numeric edits execute as network parameters. Invalid shapes, names, or non-finite values produce an error while the previous controller remains usable.

See [NEURAL_NETWORK.md](NEURAL_NETWORK.md) for the exact schema and calculations.

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

Native training supports the same readable brain JSON, edited hidden-layer sizes, resume/reset, mixed-task schedules, and live weight snapshots. Browser and PyTorch learners share task semantics, rewards, and checkpoint schema; random number streams, precision (PyTorch float32), and periodic actor refresh differ, so runs are not bit-for-bit identical. No simulated device labels are used: the UI reports the actual PyTorch device. Native training was verified end to end on CPU here; CUDA and MPS require testing on their respective hardware.

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

**Copy LLM prompt** generates a prompt for the currently selected game and the builder's architecture. **View prompt** exposes the complete text, including a manual-copy fallback if clipboard permission is unavailable. The prompt includes exact observations/scaling, action timing and tie-breaking, collisions and game rules, network equations, schema constraints, a full valid zero-filled JSON template, and the current game source. It explicitly asks the LLM to choose and write every numeric weight itself and forbids tools, internet, browsing, computer/code execution, simulation, training, optimization, or programs that generate weights. It requests only a complete literal JSON object and makes no claim of trained performance.

The default prompt architecture is **5 → 4 → 3**, only **39 parameters**, so even a small LLM has a manageable amount of literal JSON to produce. Increase the builder widths or add layers before preparing the prompt to experiment with larger models. Paste the LLM response into Add Brain, validate, name it, and load it. The app validates the model's structure and values; it cannot establish whether an external LLM actually obeyed the no-tools instruction. Fine-tuning is an explicit later step in the arena.

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

Each of the two hidden layers can contain **1–64 neurons** independently. Adding a neuron creates its incoming connections and adds a new connection to every neuron in the next layer. New outgoing weights are zero, so adding alone initially preserves the policy. Change these weights to see an effect, or train the expanded model. The five game inputs and three paddle actions are fixed interfaces.

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

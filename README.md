# Breakout Lab

A live Breakout game with keyboard controls and a tiny trainable neural network. The game and training run entirely in your browser. No Python, ML libraries, GPU, or npm packages required.

## Run on your Mac

Install [Node.js](https://nodejs.org/) 22 or newer, then from this repo:

```sh
npm start
```

Open **http://localhost:3000** in Chrome, Safari, or Firefox. No `npm install` step is needed. Stop the server with Control-C. Use `PORT=3001 npm start` if port 3000 is occupied.

## Try it

1. Click **Start playing**. Move with **← / →** or **A / D**. **Space** launches the ball or pauses/resumes. After losing a life, launch again. Touch users can drag on the playing field.
2. Click **Train neural network**. Start with a standard run; training runs in a Web Worker, so you can continue playing. The chart shows actual held-out interception error in pixels. Stop cancels the run and preserves your previous saved model.
3. When training finishes, click **Watch AI play**. This starts a fresh game controlled by the saved neural network. The dashed marker shows its predicted paddle target. Switch between **Human control** and **Neural network** at any time.
4. Weights save automatically in this browser's local storage. **Export weights** downloads a portable JSON checkpoint; **Import weights** loads it again, including on another computer. Keep an export if you clear browser data or change browser/origin.

## How it learns

This first version uses **supervised imitation learning**, not reinforcement learning. A geometric coach generates randomized ball states and ideal paddle positions, accounting for side-wall rebounds. The network learns from these examples with backpropagation and stochastic gradient descent. It receives five normalized inputs (ball x/y, ball velocity x/y, paddle x), has two 24-neuron tanh hidden layers, and predicts one target paddle position: **769 trainable parameters** total. Paddle speed is limited equally in human and AI modes.

Training is real and starts with random weights each run. A separate validation set measures average prediction error; the best checkpoint in the run is saved. During AI gameplay only the network predicts paddle targets; the geometric coach is not used. Rising balls are taught to track their current x position, and predictions update continuously as bricks change the trajectory. This is a small approximation, so missed balls are expected. A longer run can improve prediction, but validation error is not a game win rate.

## Development

```sh
npm test
```

Tests cover wall reflections, collisions, life loss, checkpoint validation, and improvement on held-out examples after actual training. `src/game.js` contains game physics, `src/network.js` the network and backpropagation, `src/trainer.js` training, `src/train-worker.js` the background worker, and `src/app.js` the live interface. `server.js` is a small Node static server.

"""Optional local PyTorch trainer: Apple MPS, NVIDIA CUDA, or CPU fallback.
Run: python3 backend/service.py (after installing backend/requirements.txt).
The browser and game simulation never require PyTorch.
"""
import copy
import json
import math
import random
import threading
import time
import uuid
from collections import deque
from environments import Environment
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import torch
    from torch import nn
except ImportError:
    raise SystemExit('Install PyTorch first: python3 -m pip install -r backend/requirements.txt')

torch.set_num_threads(2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'Apple Metal / MPS' if DEVICE == 'mps' else 'CPU fallback (no supported GPU detected)'
INPUTS = ['ball_x', 'ball_y', 'velocity_x', 'velocity_y', 'paddle_x']
OUTPUTS = ['move_left', 'stay', 'move_right']
SCHEMA = 'paddle-tracking-v1'
JOBS = {}
LOCK = threading.Lock()


def validate(request):
    plan = request.get('plan')
    if not isinstance(plan, list) or not plan or len(plan) > 2:
        raise ValueError('Choose one or two games.')
    seen = set()
    for row in plan:
        if row.get('game') not in ('breakout', 'catch') or row['game'] in seen:
            raise ValueError('Unknown or duplicate game.')
        if type(row.get('trials')) is not int or not 1 <= row['trials'] <= 100000:
            raise ValueError('Trials must be positive integers.')
        seen.add(row['game'])
    if sum(r['trials'] for r in plan) > 100000:
        raise ValueError('Maximum 100,000 total trials.')
    if request.get('order', 'interleaved') not in ('interleaved', 'sequential'):
        raise ValueError('Invalid order.')
    model = request.get('model')
    if model:
        shape = model.get('shape', [])
        if not 3 <= len(shape) <= 8 or shape[0] != 5 or shape[-1] != 3 or any(type(n) is not int or not 1 <= n <= 64 for n in shape[1:-1]):
            raise ValueError('Unsupported model shape.')
        if model.get('version') != 2 or model.get('algorithm') != 'double-dqn' or model.get('inputs') != INPUTS or model.get('outputs') != OUTPUTS or model.get('observationSchema', SCHEMA) != SCHEMA:
            raise ValueError('Incompatible brain format or observation schema.')
        # Loading validates every named connection and bias before accepting a job.
        load_model(model)
    return request


def names(layer, shape):
    return OUTPUTS if layer == len(shape)-2 else [f'hidden_{layer + 1}_{i + 1:02}' for i in range(shape[layer + 1])]


def make_model(shape):
    layers = []
    for i in range(len(shape)-1):
        layers.append(nn.Linear(shape[i], shape[i+1]))
        if i < len(shape)-2: layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def load_model(data):
    shape = data['shape']
    net = make_model(shape)
    if len(data.get('layers', [])) != len(shape)-1:
        raise ValueError('Layer count does not match shape.')
    for layer, module in enumerate([module for module in net if isinstance(module, nn.Linear)]):
        stored = data['layers'][layer]
        if stored['name'] != ('actions' if layer == len(shape)-2 else f'hidden_{layer+1}') or stored['activation'] != ('linear' if layer == len(shape)-2 else 'tanh') or len(stored['neurons']) != shape[layer + 1]:
            raise ValueError('Invalid layer structure.')
        incoming = INPUTS if layer == 0 else names(layer - 1, shape)
        rows, biases = [], []
        for i, neuron in enumerate(stored['neurons']):
            if neuron['name'] != names(layer, shape)[i] or set(neuron['weights']) != set(incoming):
                raise ValueError('Invalid neuron names or connections.')
            values = [neuron['weights'][name] for name in incoming] + [neuron['bias']]
            if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000000 for v in values):
                raise ValueError('Invalid numeric weight.')
            rows.append(values[:-1]); biases.append(values[-1])
        with torch.no_grad():
            module.weight.copy_(torch.tensor(rows)); module.bias.copy_(torch.tensor(biases))
    return net


def serialize(net, shape):
    layers = []
    for layer, module in enumerate([module for module in net if isinstance(module, nn.Linear)]):
        weights, biases = module.weight.detach().cpu().tolist(), module.bias.detach().cpu().tolist()
        incoming = INPUTS if layer == 0 else names(layer - 1, shape)
        layers.append({'name': ('actions' if layer == len(shape)-2 else f'hidden_{layer+1}'), 'activation': 'linear' if layer == len(shape)-2 else 'tanh', 'neurons': [{'name': names(layer, shape)[i], 'bias': biases[i], 'weights': dict(zip(incoming, row))} for i, row in enumerate(weights)]})
    return {'version': 2, 'algorithm': 'double-dqn', 'shape': shape, 'inputs': INPUTS, 'outputs': OUTPUTS, 'observationSchema': SCHEMA, 'layers': layers}


@torch.inference_mode()
def action(net, state):
    return int(net(torch.tensor(state, dtype=torch.float32)).argmax())


def evaluate(net, plan):
    result = {}
    for row in plan:
        scores, hits = [], []
        for i in range(5):
            env = Environment(row['game'], random.Random(9000 + i))
            for _ in range(900):
                _, done, _ = env.transition(action(net, env.observe()))
                if done: break
            scores.append(env.score); hits.append(env.hits)
        result[row['game']] = {'score': sum(scores) / 5, 'hits': sum(hits) / 5}
    return result


def run(job, request):
    try:
        rng = random.Random(2026); torch.manual_seed(2026)
        data = request.get('model'); shape = data['shape'] if data else [5, 24, 24, 3]
        resume = bool(data and not request.get('resetWeights'))
        online = (load_model(data) if resume else make_model(shape)).to(DEVICE)
        target = copy.deepcopy(online).eval()
        actor = copy.deepcopy(online).to('cpu').eval()
        optimizer = torch.optim.Adam(online.parameters(), lr=.0006)
        replay = deque(maxlen=30000)
        plan = request['plan']; order = request.get('order', 'interleaved')
        trials = [r['game'] for i in range(max(r['trials'] for r in plan)) for r in plan if i < r['trials']] if order == 'interleaved' else [r['game'] for r in plan for _ in range(r['trials'])]
        completed = {r['game']: 0 for r in plan}
        by_game = evaluate(actor, plan)
        best = sum(v['score'] for v in by_game.values()) / len(by_game)
        checkpoint = serialize(actor, shape)
        steps = updates = frames = 0
        for episode, game in enumerate(trials, 1):
            if job['cancel'].is_set(): job['status'] = 'cancelled'; return
            env = Environment(game, rng)
            for _ in range(900):
                if job['cancel'].is_set(): job['status'] = 'cancelled'; return
                state = env.observe(); epsilon = .03 + (.17 if resume else .97) * math.exp(-steps / 25000)
                chosen = rng.randrange(3) if rng.random() < epsilon else action(actor, state)
                reward, done, count = env.transition(chosen)
                replay.append((state, chosen, reward, env.observe(), done)); frames += count; steps += 1
                if len(replay) >= 512 and steps % 4 == 0:
                    batch = rng.sample(list(replay), 32)
                    states = torch.tensor([s[0] for s in batch], device=DEVICE, dtype=torch.float32)
                    actions = torch.tensor([s[1] for s in batch], device=DEVICE)
                    rewards = torch.tensor([s[2] for s in batch], device=DEVICE, dtype=torch.float32)
                    next_states = torch.tensor([s[3] for s in batch], device=DEVICE, dtype=torch.float32)
                    terminal = torch.tensor([s[4] for s in batch], device=DEVICE, dtype=torch.float32)
                    with torch.no_grad():
                        next_actions = online(next_states).argmax(1)
                        values = rewards + .99 * (1 - terminal) * target(next_states).gather(1, next_actions[:, None]).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(online(states).gather(1, actions[:, None]).squeeze(1), values)
                    optimizer.zero_grad(); loss.backward(); optimizer.step(); updates += 1
                    if updates % 64 == 0: actor.load_state_dict({k: v.detach().cpu() for k, v in online.state_dict().items()})
                    if updates % 400 == 0: target.load_state_dict(online.state_dict())
                if done: break
            completed[game] += 1
            if episode % 100 == 0 or episode == len(trials):
                actor.load_state_dict({k: v.detach().cpu() for k, v in online.state_dict().items()})
                by_game = evaluate(actor, plan)
                score = sum(v['score'] for v in by_game.values()) / len(by_game)
                if score > best:
                    best = score; checkpoint = serialize(actor, shape)
                    saved = dict(checkpoint, score=best, device=DEVICE, trainingPlan=plan, order=order, completedTrials=dict(completed), episodes=episode, frames=frames)
                    job.update(checkpoint=saved, checkpointEpisode=episode)
            if episode % 10 == 0 or episode == len(trials):
                snapshot = serialize(online, shape); snapshot['visualizationInput'] = state
                job.update(episode=episode, frames=frames, updates=updates, byGame=by_game, snapshot=snapshot)
        checkpoint.update(score=best, episodes=len(trials), frames=frames, device=DEVICE, trainingPlan=plan, order=order, completedTrials=completed, trainingHistory=(data or {}).get('trainingHistory', []) + [{'plan': plan, 'order': order, 'completed': completed, 'device': DEVICE}])
        job.update(status='complete', model=checkpoint)
    except Exception as error:
        job.update(status='error', error=str(error))


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data):
        payload = json.dumps(data).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/status': return self.respond(200, {'device': DEVICE, 'name': DEVICE_NAME})
        with LOCK: job = JOBS.get(self.path.removeprefix('/jobs/'))
        if not job: return self.respond(404, {'error': 'Job not found.'})
        self.respond(200, {k: v for k, v in job.items() if k != 'cancel'})

    def do_POST(self):
        if self.path != '/jobs': return self.respond(404, {'error': 'Not found.'})
        try:
            size = int(self.headers.get('Content-Length', 0))
            if size < 1 or size > 1500000: raise ValueError('Invalid request size.')
            request = validate(json.loads(self.rfile.read(size)))
            with LOCK:
                if any(j['status'] == 'running' for j in JOBS.values()): return self.respond(409, {'error': 'A native training job is already running.'})
                if len(JOBS) >= 16: JOBS.pop(next(iter(JOBS)))
                ident = uuid.uuid4().hex; job = {'id': ident, 'status': 'running', 'device': DEVICE, 'cancel': threading.Event(), 'episode': 0}; JOBS[ident] = job
            threading.Thread(target=run, args=(job, request), daemon=True).start()
            self.respond(202, {'id': ident, 'device': DEVICE})
        except Exception as error: self.respond(400, {'error': str(error)})

    def do_DELETE(self):
        with LOCK: job = JOBS.get(self.path.removeprefix('/jobs/'))
        if not job: return self.respond(404, {'error': 'Job not found.'})
        job['cancel'].set(); self.respond(200, {'status': 'stopping'})


if __name__ == '__main__':
    print(f'Native trainer on http://127.0.0.1:3001 · {DEVICE}: {DEVICE_NAME}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 3001), Handler).serve_forever()

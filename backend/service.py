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
from environments import Environment, validate_rewards
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
    request['rewardConfig'] = validate_rewards(request.get('rewardConfig', (model or {}).get('rewardConfig')))
    if request.get('checkpointSelection', 'score') not in ('score', 'latest'): raise ValueError('Choose score or latest checkpoint selection.')
    choose_device(request.get('device', 'auto'))
    if request.get('mode','automatic') not in ('automatic','manual'): raise ValueError('Invalid training mode.')
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


def parameter_vector(net):
    # One device-to-host transfer, rather than one sync per weight/bias tensor.
    return torch.nn.utils.parameters_to_vector(net.parameters()).detach().cpu()


def refresh_actor(actor, online):
    torch.nn.utils.vector_to_parameters(parameter_vector(online), actor.parameters())


def serialize(net, shape):
    flat = parameter_vector(net).tolist()
    cursor = 0
    layers = []
    for layer in range(len(shape)-1):
        width, inputs = shape[layer+1], shape[layer]
        values = flat[cursor:cursor+width*inputs]; cursor += width*inputs
        biases = flat[cursor:cursor+width]; cursor += width
        incoming = INPUTS if layer == 0 else names(layer-1, shape)
        layers.append({'name': 'actions' if layer == len(shape)-2 else f'hidden_{layer+1}', 'activation': 'linear' if layer == len(shape)-2 else 'tanh', 'neurons': [{'name': names(layer, shape)[i], 'bias': biases[i], 'weights': dict(zip(incoming, values[i*inputs:(i+1)*inputs]))} for i in range(width)]})
    return {'version': 2, 'algorithm': 'double-dqn', 'shape': shape, 'inputs': INPUTS, 'outputs': OUTPUTS, 'observationSchema': SCHEMA, 'layers': layers}


def choose_device(requested='auto'):
    if requested == 'auto': return DEVICE
    if requested == 'cpu': return 'cpu'
    if requested == 'cuda' and torch.cuda.is_available(): return 'cuda'
    if requested == 'mps' and torch.backends.mps.is_available(): return 'mps'
    raise ValueError(f'Requested device {requested} is unavailable.')


def training_profile(device):
    # Keep 8 replay samples per new transition. GPUs do 16x fewer launches;
    # actor/target refreshes remain tied to environment transitions, not batches.
    return {'batchSize': 512, 'updateEvery': 64} if device != 'cpu' else {'batchSize': 32, 'updateEvery': 4}


class ReplayBuffer:
    def __init__(self, capacity=30000): self.items = []; self.capacity = capacity; self.cursor = 0
    def append(self, row):
        if len(self.items) < self.capacity: self.items.append(row)
        else: self.items[self.cursor] = row
        self.cursor = (self.cursor+1) % self.capacity
    def __len__(self): return len(self.items)
    def sample(self, rng, count): return rng.sample(self.items, count)


def learn_batch(online, target, optimizer, replay, rng, count, device):
    # Pack all transition fields into a single transfer (13 floats per sample).
    packed = torch.tensor(replay.sample(rng, count), dtype=torch.float32)
    if device == 'cuda': packed = packed.pin_memory()
    packed = packed.to(device, non_blocking=device != 'cpu')
    states, actions, rewards = packed[:, :5], packed[:, 5].long(), packed[:, 6]
    next_states, terminal = packed[:, 7:12], packed[:, 12]
    with torch.no_grad():
        next_actions = online(next_states).argmax(1)
        values = rewards + .99 * (1-terminal) * target(next_states).gather(1, next_actions[:, None]).squeeze(1)
    loss = nn.functional.smooth_l1_loss(online(states).gather(1, actions[:, None]).squeeze(1), values)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    # Deliberately do not .item() the GPU loss: that would force a sync per batch.


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
        rewards = validate_rewards(request.get('rewardConfig', (request.get('model') or {}).get('rewardConfig')))
        selection = request.get('checkpointSelection', 'score')
        if selection not in ('score', 'latest'): raise ValueError('Choose score or latest checkpoint selection.')
        metadata = dict(rewardConfig=rewards, checkpointSelection=selection, discountFactor=.99, trainingSeed=2026, evaluationSeeds=list(range(9000,9005)))
        recent = []
        checkpoint_episode = 0
        device = choose_device(request.get('device','auto')); profile = training_profile(device)
        job.setdefault("control", threading.Condition()); job.setdefault("mode", request.get("mode", "automatic")); job.setdefault("permits", 1 if job["mode"]=="manual" else 0)
        batch_size, update_every = profile['batchSize'], profile['updateEvery']
        job.update(device=device, profile=profile, phase='Preparing training')
        started = time.perf_counter(); last_snapshot = started
        rng = random.Random(2026); torch.manual_seed(2026)
        data = request.get('model'); shape = data['shape'] if data else [5, 24, 24, 3]
        resume = bool(data and not request.get('resetWeights'))
        online = (load_model(data) if resume else make_model(shape)).to(device)
        target = copy.deepcopy(online).eval()
        actor = copy.deepcopy(online).to('cpu').eval()
        optimizer = torch.optim.Adam(online.parameters(), lr=.0006)
        replay = ReplayBuffer()
        plan = request['plan']; order = request.get('order', 'interleaved')
        trials = [r['game'] for i in range(max(r['trials'] for r in plan)) for r in plan if i < r['trials']] if order == 'interleaved' else [r['game'] for r in plan for _ in range(r['trials'])]
        completed = {r['game']: 0 for r in plan}
        by_game = evaluate(actor, plan)
        best = sum(v['score'] for v in by_game.values()) / len(by_game)
        best_hits=sum(v['hits'] for v in by_game.values())
        job.update(byGame=by_game,evaluationEpisode=0)
        checkpoint = serialize(actor, shape)
        steps = updates = frames = 0
        actor_refresh = target_refresh = 0
        for episode, game in enumerate(trials, 1):
            with job['control']:
                while job['mode']=='manual' and job['permits']==0 and not job['cancel'].is_set():
                    job['phase']='Waiting for Forward'; job['control'].wait(.1)
                if job['mode']=='manual' and job['permits']>0: job['permits']-=1
            if job['cancel'].is_set(): job['status'] = 'cancelled'; return
            env = Environment(game, rng, rewards)
            episode_terms = dict.fromkeys(rewards, 0)
            episode_reward = 0
            for _ in range(900):
                if job['cancel'].is_set(): job['status'] = 'cancelled'; return
                state = env.observe(); epsilon = .03 + (.17 if resume else .97) * math.exp(-steps / 25000)
                chosen = rng.randrange(3) if rng.random() < epsilon else action(actor, state)
                reward, done, count = env.transition(chosen)
                episode_reward += reward
                for key, value in env.last_reward_terms.items(): episode_terms[key] += value
                replay.append([*state, chosen, reward, *env.observe(), float(done)]); frames += count; steps += 1
                manual = job['mode']=='manual'
                active_profile = {'batchSize':32,'updateEvery':4} if manual else profile
                batch_size, update_every = active_profile['batchSize'], active_profile['updateEvery']
                job['profile']=active_profile
                if len(replay) >= (32 if manual else max(512, batch_size)) and steps % update_every == 0:
                    learn_batch(online, target, optimizer, replay, rng, batch_size, device); updates += 1
                    if steps-actor_refresh >= 256: refresh_actor(actor, online); actor_refresh = steps
                    if steps-target_refresh >= 1600: target.load_state_dict(online.state_dict()); target_refresh = steps
                if done: break
            completed[game] += 1
            recent.append((episode_reward, episode_terms))
            recent = recent[-100:]
            job.update(reward=sum(item[0] for item in recent)/len(recent), rewardTerms={key:sum(item[1][key] for item in recent)/len(recent) for key in rewards})
            if episode % 100 == 0 or episode == len(trials) or job["mode"]=="manual":
                refresh_actor(actor, online)
                by_game = evaluate(actor, plan)
                score = sum(v['score'] for v in by_game.values()) / len(by_game)
                hits=sum(v['hits'] for v in by_game.values())
                job.update(byGame=by_game,evaluationEpisode=episode)
                if selection == "latest" or score > best or (score==best and hits>=best_hits):
                    best_hits=hits;best = score; checkpoint = serialize(actor, shape)
                    checkpoint_episode = episode
                    saved = dict(checkpoint, **metadata, checkpointEpisode=episode, score=best, device=device, trainingProfile=profile, trainingPlan=plan, order=order, completedTrials=dict(completed), episodes=episode, frames=frames)
                    job.update(checkpoint=saved, checkpointEpisode=episode)
            elapsed = time.perf_counter()-started
            job.update(episode=episode, frames=frames, updates=updates, replaySize=len(replay), framesPerSecond=round(frames/max(.001,elapsed)), elapsedSeconds=elapsed, byGame=by_game, phase='Waiting for Forward' if job['mode']=='manual' else 'Training')
            if (episode % 10 == 0 and time.perf_counter()-last_snapshot >= 1) or episode == len(trials) or job['mode']=='manual':
                snapshot = serialize(online, shape); snapshot['visualizationInput'] = state
                job.update(snapshot=snapshot); last_snapshot=time.perf_counter()
                if job['mode']=='manual': job['manualModel']=dict(snapshot, **metadata, checkpointEpisode=episode, episodes=episode, frames=frames, completedTrials=dict(completed), trainingPlan=plan, order=order, device=device,score=sum(v['score'] for v in by_game.values())/len(by_game))
        checkpoint.update(**metadata, checkpointEpisode=checkpoint_episode, score=best, episodes=len(trials), frames=frames, device=device, trainingProfile=profile, trainingPlan=plan, order=order, completedTrials=completed, trainingHistory=(data or {}).get('trainingHistory', []) + [{'plan': plan, 'order': order, 'completed': completed, 'device': device}])
        job.update(status='complete', model=job.get('manualModel',checkpoint) if job['mode']=='manual' else checkpoint)
    except Exception as error:
        job.update(status='error', error=str(error))


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data):
        payload = json.dumps(data).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

    def do_GET(self):
        if self.path == '/status': return self.respond(200, {'device': DEVICE, 'name': DEVICE_NAME, 'profile': training_profile(DEVICE)})
        with LOCK: job = JOBS.get(self.path.removeprefix('/jobs/'))
        if not job: return self.respond(404, {'error': 'Job not found.'})
        self.respond(200, {k: v for k, v in job.items() if k not in ('cancel','control')})

    def do_POST(self):
        if self.path.startswith('/jobs/'):
            parts=self.path.split('/')
            with LOCK: job=JOBS.get(parts[2])
            if not job or len(parts)!=4 or parts[3] not in ('step','run','pause'): return self.respond(404, {'error':'Training session not found.'})
            with job['control']:
                job['mode']='automatic' if parts[3]=='run' else 'manual'
                job['permits']=job.get('permits',0)+1 if parts[3]=='step' else 0
                job['control'].notify_all()
            return self.respond(200, {'mode':job['mode']})
        if self.path != '/jobs': return self.respond(404, {'error': 'Not found.'})
        try:
            size = int(self.headers.get('Content-Length', 0))
            if size < 1 or size > 1500000: raise ValueError('Invalid request size.')
            request = validate(json.loads(self.rfile.read(size)))
            with LOCK:
                if any(j['status'] == 'running' for j in JOBS.values()): return self.respond(409, {'error': 'A native training job is already running.'})
                if len(JOBS) >= 16: JOBS.pop(next(iter(JOBS)))
                ident = uuid.uuid4().hex; job = {'id': ident, 'status': 'running', 'device': choose_device(request.get('device','auto')), 'cancel': threading.Event(), 'control':threading.Condition(), 'mode':request.get('mode','automatic'), 'permits':1 if request.get('mode')=='manual' else 0, 'episode': 0}; JOBS[ident] = job
            threading.Thread(target=run, args=(job, request), daemon=True).start()
            self.respond(202, {'id': ident, 'device': job['device']})
        except Exception as error: self.respond(400, {'error': str(error)})

    def do_DELETE(self):
        with LOCK: job = JOBS.get(self.path.removeprefix('/jobs/'))
        if not job: return self.respond(404, {'error': 'Job not found.'})
        job['cancel'].set(); self.respond(200, {'status': 'stopping'})


if __name__ == '__main__':
    print(f'Native trainer on http://127.0.0.1:3001 · {DEVICE}: {DEVICE_NAME}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 3001), Handler).serve_forever()

"""Measure real native learner throughput on this machine (with GPU synchronization).
Run: python3 backend/benchmark.py --device all --samples 16384
This is a diagnostic, not a game score or a promise of faster learning.
"""
import argparse
import copy
import json
import random
import time
import torch
import service


def synchronize(device):
    if device == 'mps': torch.mps.synchronize()
    elif device == 'cuda': torch.cuda.synchronize()


def measure(device, batch, samples, shape):
    torch.manual_seed(42)
    online = service.make_model(shape).to(device)
    target = copy.deepcopy(online).eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=.0006)
    replay = service.ReplayBuffer(); rng = random.Random(42)
    for _ in range(4096):
        replay.append([*[rng.uniform(-1, 1) for _ in range(5)], rng.randrange(3), rng.uniform(-1, 1), *[rng.uniform(-1, 1) for _ in range(5)], float(rng.random() < .05)])
    # Warm up kernels/Metal shaders and optimizer state outside the timed region.
    for _ in range(10): service.learn_batch(online, target, optimizer, replay, rng, batch, device)
    synchronize(device); started = time.perf_counter()
    updates = max(1, samples // batch)
    for _ in range(updates): service.learn_batch(online, target, optimizer, replay, rng, batch, device)
    synchronize(device); elapsed = time.perf_counter()-started
    return {'device': device, 'batchSize': batch, 'updates': updates, 'samples': updates*batch, 'seconds': round(elapsed, 4), 'samplesPerSecond': round(updates*batch/elapsed)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['all','auto','cpu','mps','cuda'], default='all')
    parser.add_argument('--samples', type=int, default=16384)
    parser.add_argument('--hidden', default='24,24')
    args = parser.parse_args()
    if args.samples < 512: parser.error('Use at least 512 samples.')
    shape=[5,*map(int,args.hidden.split(',')),3]
    service.validate({'plan':[{'game':'catch','trials':1}],'model':service.serialize(service.make_model(shape),shape)})
    devices=['cpu']+(['mps'] if torch.backends.mps.is_available() else [])+(['cuda'] if torch.cuda.is_available() else []) if args.device=='all' else [service.choose_device(args.device)]
    print(json.dumps({'shape':shape,'note':'Same sample budget; 512 batches perform fewer optimizer updates. Includes sampling, transfers and synchronized compute, excludes game physics.','results':[measure(d,b,args.samples,shape) for d in devices for b in [32,512]]},indent=2))

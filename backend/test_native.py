import json
import random
import unittest
from environments import Environment


class PhysicsTests(unittest.TestCase):
    def test_catch_and_miss(self):
        env = Environment('catch', random.Random(1))
        env.x, env.y, env.vx, env.vy = env.paddle, 480, 0, 4
        reward, done, _ = env.transition(1)
        self.assertEqual(env.hits, 1)
        self.assertEqual(env.score, 10)
        self.assertFalse(done)
        env.x, env.y = 10, 550
        self.assertTrue(env.transition(1)[1])

    def test_breakout_hit(self):
        env = Environment('breakout', random.Random(1))
        env.x, env.y, env.vx, env.vy = env.paddle, 480, 0, 4
        env.transition(1)
        self.assertEqual(env.hits, 1)
        self.assertLess(env.vy, 0)


try:
    import service
except (ImportError, SystemExit):
    service = None


@unittest.skipIf(service is None, 'Optional PyTorch not installed')
class NativeTests(unittest.TestCase):
    def test_readable_checkpoint_roundtrip_and_validation(self):
        import torch
        model = service.make_model([5, 3, 4, 3])
        data = service.serialize(model, [5, 3, 4, 3])
        loaded = service.load_model(data)
        sample = torch.tensor([.1, .2, .3, .4, .5])
        self.assertTrue(torch.equal(model(sample), loaded(sample)))
        data['layers'][0]['neurons'][0]['weights']['ball_x'] = None
        with self.assertRaises(ValueError): service.load_model(data)

    def test_extra_layers_roundtrip(self):
        for shape in ([5, 4, 3], [5, 2, 3, 2, 3]):
            serialized = service.serialize(service.make_model(shape), shape)
            self.assertEqual(service.serialize(service.load_model(serialized), shape)["shape"], shape)

    def test_native_mixed_training_optimizes_and_returns_compatible_brain(self):
        import threading
        job = {'cancel': threading.Event(), 'status': 'running'}
        model = service.serialize(service.make_model([5, 3, 4, 3]), [5, 3, 4, 3])
        request = service.validate({'model': model, 'plan': [{'game': 'breakout', 'trials': 12}, {'game': 'catch', 'trials': 12}], 'order': 'interleaved'})
        service.run(job, request)
        self.assertEqual(job['status'], 'complete', job.get('error'))
        self.assertEqual(job['model']['completedTrials'], {'breakout': 12, 'catch': 12})
        service.load_model(job['model'])
        self.assertGreater(job['frames'], 0)
        self.assertGreater(job['updates'], 0)


if __name__ == '__main__': unittest.main()

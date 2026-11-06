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

    def test_gpu_profile_and_packed_replay_update(self):
        import copy
        import torch
        self.assertEqual(service.training_profile('mps'), {'batchSize':512,'updateEvery':64})
        replay=service.ReplayBuffer(capacity=600)
        for _ in range(700): replay.append([.1,.2,.3,.4,.5,2,1,.2,.3,.4,.5,.6,0])
        self.assertEqual(len(replay),600)
        online=service.make_model([5,3,3]);target=copy.deepcopy(online)
        before=service.parameter_vector(online).clone()
        service.learn_batch(online,target,torch.optim.Adam(online.parameters(),lr=.001),replay,random.Random(1),512,'cpu')
        self.assertFalse(torch.equal(before,service.parameter_vector(online)))

    def test_manual_session_waits_for_each_forward(self):
        import threading
        import time
        job={'cancel':threading.Event(),'status':'running'}
        model=service.serialize(service.make_model([5,3,3]),[5,3,3])
        request={'model':model,'mode':'manual','plan':[{'game':'breakout','trials':3}],'order':'interleaved'}
        thread=threading.Thread(target=service.run,args=(job,request),daemon=True);thread.start()
        def wait_episode(number):
            deadline=time.monotonic()+10
            while job.get('manualModel',{}).get('episodes',0)<number and time.monotonic()<deadline and job.get('status')!='error':time.sleep(.01)
            self.assertEqual(job.get('manualModel',{}).get('episodes'),number,job.get('error'))
        try:
            wait_episode(1);time.sleep(.05);self.assertEqual(job['episode'],1)
            with job['control']:job['permits']+=1;job['control'].notify_all()
            wait_episode(2)
            with job['control']:job['mode']='automatic';job['control'].notify_all()
            thread.join(10);self.assertFalse(thread.is_alive());self.assertEqual(job['status'],'complete')
        finally:job['cancel'].set()

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

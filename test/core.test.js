import test from 'node:test';
import assert from 'node:assert/strict';
import {Game,coach,reflect,PADDLE_Y,R} from '../src/game.js';
import {Network,rng} from '../src/network.js';
import {train,sample,evaluate} from '../src/trainer.js';
test('reflections handle multiple wall bounces and coach predicts intercept',()=>{assert.equal(reflect(903),883);assert.equal(reflect(-3),17);assert.equal(reflect(7+886*4+100),107);assert.equal(coach({x:400,y:400,vx:2,vy:4}),440.5);});
test('paddle collision, brick collision and lost life',()=>{const g=new Game(rng());g.launch();g.ball={x:450,y:PADDLE_Y-R-1,vx:0,vy:4};g.step(1);assert.ok(g.ball.vy<0);assert.equal(g.hits,1);const b=g.bricks[40];g.ball={x:b.x+20,y:b.y+b.h+R+1,vx:0,vy:-4};g.step(1);assert.equal(b.alive,false);assert.equal(g.score,10);g.ball.y=560;g.step(1);assert.equal(g.lives,2);assert.equal(g.state,'ready');});
test('model roundtrip and malformed weight rejection',()=>{const n=new Network(null,rng());const a=[.1,.2,.3,.4,.5];assert.equal(Network.load(JSON.parse(JSON.stringify(n.serialize()))).predict(a),n.predict(a));assert.throws(()=>Network.load({version:1,shape:[5,24,24,1],weights:[]}));const bad=n.serialize();bad.weights[0][0][0]=NaN;assert.throws(()=>Network.load(bad));});
test('real training reduces held-out interception error',()=>{const random=rng(990),validation=Array.from({length:600},()=>sample(random)),initial=evaluate(new Network(null,rng(2026)),validation);const model=train({epochs:150});const error=evaluate(Network.load(model),validation);console.log(`Held-out error: ${initial.toFixed(1)} → ${error.toFixed(1)} pixels`);assert.ok(error<initial*.4);assert.ok(error<40);});

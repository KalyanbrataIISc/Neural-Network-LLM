export const W=900,H=540,PADDLE_Y=488,PADDLE_W=116,R=7;
export const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export function reflect(x,lo=R,hi=W-R) { const span=hi-lo; let v=((x-lo)%(span*2)+span*2)%(span*2); return lo+(v>span?span*2-v:v); }
export function inputs(ball,paddle) { return [ball.x/W*2-1,ball.y/H*2-1,ball.vx/6,ball.vy/6,paddle/W*2-1]; }
// The coach is used ONLY to generate training labels. AI gameplay uses network.predict.
export function coach(ball) { return ball.vy>0 ? clamp(reflect(ball.x+ball.vx*Math.max(0,(PADDLE_Y-R-ball.y)/ball.vy)),PADDLE_W/2,W-PADDLE_W/2) : clamp(ball.x,PADDLE_W/2,W-PADDLE_W/2); }
export class Game {
  constructor(random=Math.random){this.random=random;this.reset();}
  reset(){this.paddle=W/2;this.score=0;this.lives=3;this.hits=0;this.state='ready';this.bricks=Array.from({length:50},(_,i)=>({x:37+i%10*83,y:55+Math.floor(i/10)*30,w:76,h:21,row:Math.floor(i/10),alive:true}));this.serve();}
  serve(){this.ball={x:this.paddle,y:PADDLE_Y-14,vx:(this.random()>.5?1:-1)*(2+this.random()),vy:-4.1};this.trail=[];}
  launch(){if(this.state==='ready'){this.state='playing';return;} if(this.state==='paused')this.state='playing';else if(this.state==='playing')this.state='paused';}
  step(dt, movement=0,target=null){
    if(this.state!=='playing' && this.state!=='ready')return;
    this.paddle=clamp(this.paddle+(target===null?movement*8*dt:clamp(target-this.paddle,-8*dt,8*dt)),PADDLE_W/2,W-PADDLE_W/2);
    if(this.state==='ready'){this.ball.x=this.paddle;return;}
    const b=this.ball, oldX=b.x, oldY=b.y;
    b.x+=b.vx*dt;b.y+=b.vy*dt;
    if(b.x<R){b.x=R;b.vx=Math.abs(b.vx);}if(b.x>W-R){b.x=W-R;b.vx=-Math.abs(b.vx);}if(b.y<R){b.y=R;b.vy=Math.abs(b.vy);}
    if(b.vy>0 && oldY+R<=PADDLE_Y && b.y+R>=PADDLE_Y && Math.abs(b.x-this.paddle)<PADDLE_W/2+R){
      b.y=PADDLE_Y-R;const offset=clamp((b.x-this.paddle)/(PADDLE_W/2),-1,1);b.vx=offset*4.3;b.vy=-Math.sqrt(5.5**2-b.vx**2);this.hits++;
    }
    for(const brick of this.bricks){if(!brick.alive)continue;if(b.x+R>brick.x&&b.x-R<brick.x+brick.w&&b.y+R>brick.y&&b.y-R<brick.y+brick.h){brick.alive=false;this.score+=(5-brick.row)*10;if(oldY+R<=brick.y){b.y=brick.y-R;b.vy=-Math.abs(b.vy);}else if(oldY-R>=brick.y+brick.h){b.y=brick.y+brick.h+R;b.vy=Math.abs(b.vy);}else {b.x=oldX; b.vx*=-1;}break;}}
    this.trail.push({x:b.x,y:b.y});if(this.trail.length>12)this.trail.shift();
    if(b.y>H+R){this.lives--;this.state=this.lives?'ready':'lost';this.serve();}
    if(this.bricks.every(b=>!b.alive))this.state='won';
  }
}

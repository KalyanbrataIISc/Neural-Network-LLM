"""CPU game simulation shared by the optional native training service."""
import math

class Environment:
    """Same dimensions, observations, actions, rewards and collisions as the JS tasks."""
    def __init__(self, game, rng):
        self.game, self.rng = game, rng
        self.paddle = 80 + rng.random() * 740
        self.score = self.hits = 0
        self.done = False
        self.bricks = [[37 + i % 10 * 83, 55 + i // 10 * 30, i // 10, True] for i in range(50)] if game == 'breakout' else []
        self.serve()

    def serve(self):
        r = self.rng
        self.x, self.y, self.vx, self.vy = (30 + r.random() * 840, 25, (r.random() * 2 - 1) * 2.5, 3.5 + r.random() * 1.5) if self.game == 'catch' else (self.paddle, 474, (1 if r.random() > .5 else -1) * (2 + r.random()), -4.1)

    def observe(self):
        return [self.x / 900 * 2 - 1, self.y / 540 * 2 - 1, self.vx / 6, self.vy / 6, self.paddle / 900 * 2 - 1]

    def transition(self, action):
        old_score, old_hits = self.score, self.hits
        phi = -abs(self.x - self.paddle) / 100
        frames = 0
        for _ in range(4):
            frames += 1
            self.paddle = max(58, min(842, self.paddle + (action - 1) * 8))
            ox, oy = self.x, self.y
            self.x += self.vx; self.y += self.vy
            if self.x < 7: self.x, self.vx = 7, abs(self.vx)
            if self.x > 893: self.x, self.vx = 893, -abs(self.vx)
            if self.y < 7: self.y, self.vy = 7, abs(self.vy)
            if self.vy > 0 and oy + 7 <= 488 <= self.y + 7 and abs(self.x - self.paddle) < 65:
                self.hits += 1
                if self.game == 'catch':
                    self.score += 10; self.serve(); continue
                self.y = 481
                self.vx = max(-1, min(1, (self.x - self.paddle) / 58)) * 4.3
                self.vy = -math.sqrt(5.5 ** 2 - self.vx ** 2)
            for brick in self.bricks:
                x, y, row, alive = brick
                if alive and self.x + 7 > x and self.x - 7 < x + 76 and self.y + 7 > y and self.y - 7 < y + 21:
                    brick[3] = False; self.score += (5 - row) * 10
                    if oy + 7 <= y: self.y, self.vy = y - 7, -abs(self.vy)
                    elif oy - 7 >= y + 21: self.y, self.vy = y + 28, abs(self.vy)
                    else: self.x, self.vx = ox, -self.vx
                    break
            if self.y > 547:
                self.done = True; break
            if self.bricks and not any(b[3] for b in self.bricks):
                self.done = True; break
        won = self.game == 'breakout' and not any(b[3] for b in self.bricks)
        reward = (self.score - old_score) * .025 + (self.hits - old_hits) * 2 - (3 if self.done and not won else 0) + (10 if won else 0) + .8 * ((0 if self.done else .99 * -abs(self.x - self.paddle) / 100) - phi) - .025 * (abs(self.x - self.paddle) / 900 if self.vy > 0 else 0)
        return reward, self.done, frames



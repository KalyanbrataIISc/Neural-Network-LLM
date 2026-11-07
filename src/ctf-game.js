import { rng } from "./network.js";
import { validateRewards, defaultRewards } from "./rewards.js";

import { generateMap, sampleMapId, MAP_PROTOCOL, MAP_SPLITS, LEGACY_MAP, MAP_WIDTH, MAP_HEIGHT } from "./ctf-maps.js";
export const WIDTH = MAP_WIDTH, HEIGHT = MAP_HEIGHT, RADIUS = 12;
export const DIRECTIONS = [
  [0, -1], [Math.SQRT1_2, -Math.SQRT1_2], [1, 0],
  [Math.SQRT1_2, Math.SQRT1_2], [0, 1],
  [-Math.SQRT1_2, Math.SQRT1_2], [-1, 0], [-Math.SQRT1_2, -Math.SQRT1_2],
];
export const DIRECTION_NAMES = ["north", "north_east", "east", "south_east", "south", "south_west", "west", "north_west"];
export const ABILITIES = ["move", "fire", "dash"];
const SELF_INPUTS = ["x", "y", "velocity_x", "velocity_y", "health", "energy", "ammo", "fire_cooldown", "dash_cooldown", "carrying_flag", "flag_dx", "flag_dy", "base_dx", "base_dy", "facing_x", "facing_y"];
export const CTF_INPUT_NAMES = [
  ...SELF_INPUTS,
  ...DIRECTION_NAMES.flatMap((direction) => ["wall", "enemy", "supply", "bullet"].map((kind) => `${direction}_${kind}_range`)),
  ...Array.from({ length: 4 }, (_, i) => ["dx", "dy", "vx", "vy", "health", "range", "visible", "alive"].map((name) => `enemy_${i + 1}_${name}`)).flat(),
  ...Array.from({ length: 4 }, (_, i) => ["dx", "dy", "type", "available"].map((name) => `supply_${i + 1}_${name}`)).flat(),
];
export const CTF_OUTPUT_NAMES = ABILITIES.flatMap((ability) => DIRECTION_NAMES.map((direction) => `${ability}_${direction}`));
export const CTF_SPEC = Object.freeze({ inputs: CTF_INPUT_NAMES, outputs: CTF_OUTPUT_NAMES, minHiddenLayers: 1, maxHiddenLayers: 4, maxHiddenNeurons: 128, observationSchema: "robot-capture-v1" });
export const CTF_REWARDS = [
  { key: "flagPickup", label: "Pick up the flag", value: 8, description: "Added once when you reach the enemy flag." },
  { key: "delivery", label: "Deliver the flag", value: 30, description: "Added when you bring the flag to your base. Wins and ends the episode." },
  { key: "enemyHit", label: "Hit an enemy", value: 1, description: "Added per projectile hit on an enemy (20 damage)." },
  { key: "enemyDefeat", label: "Defeat an enemy", value: 3, description: "Added when an enemy's health reaches zero, in addition to the hit reward." },
  { key: "damage", label: "Per health point lost", value: -0.15, description: "Multiply health lost from enemy projectiles by this value." },
  { key: "supply", label: "Collect useful supplies", value: 0.5, description: "Added for each collected health, ammo, or energy pack. Full resources cannot collect that pack." },
  { key: "progress", label: "Objective progress", value: 2, description: "Multiply (distance before − distance after) / 100. Distances are in pixels, to the flag or, when carrying, your base. The target is fixed for the duration of the action." },
  { key: "defeat", label: "Robot destroyed", value: -10, description: "Added at zero health. Ends the episode." },
  { key: "step", label: "Time cost per action", value: -0.002, description: "Added once per action (up to 4 physics frames), to encourage timely delivery." },
];
export const CTF_DEFAULT_REWARDS = Object.freeze(defaultRewards(CTF_REWARDS));
export const WALLS = LEGACY_MAP.walls, BASE = LEGACY_MAP.base, FLAG = LEGACY_MAP.flag;
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
function blocked(x, y, radius = RADIUS, walls = WALLS) {
  return x < radius || y < radius || x > WIDTH - radius || y > HEIGHT - radius ||
    walls.some((w) => x + radius > w.x && x - radius < w.x + w.w && y + radius > w.y && y - radius < w.y + w.h);
}
function rayBoxDistance(origin, dx, dy, wall) {
  let near = 0, far = Infinity;
  for (const [position, direction, low, high] of [
    [origin.x, dx, wall.x, wall.x + wall.w],
    [origin.y, dy, wall.y, wall.y + wall.h],
  ]) {
    if (Math.abs(direction) < 1e-12) {
      if (position < low || position > high) return Infinity;
    } else {
      const a = (low - position) / direction, b = (high - position) / direction;
      near = Math.max(near, Math.min(a, b)); far = Math.min(far, Math.max(a, b));
      if (near > far) return Infinity;
    }
  }
  return far < 0 ? Infinity : near;
}
export function visible(a, b, walls = WALLS) {
  return walls.every((wall) => rayBoxDistance(a, b.x - a.x, b.y - a.y, wall) >= 1);
}
function move(entity, dx, dy, walls) {
  const oldX = entity.x, oldY = entity.y;
  if (!blocked(entity.x + dx, entity.y, RADIUS, walls)) entity.x += dx;
  if (!blocked(entity.x, entity.y + dy, RADIUS, walls)) entity.y += dy;
  entity.vx = entity.x - oldX;
  entity.vy = entity.y - oldY;
}
function rayRange(player, direction, walls) {
  const [dx, dy] = direction;
  let nearest = Math.min(
    dx > 0 ? (WIDTH - player.x) / dx : dx < 0 ? -player.x / dx : Infinity,
    dy > 0 ? (HEIGHT - player.y) / dy : dy < 0 ? -player.y / dy : Infinity,
  );
  for (const wall of walls) nearest = Math.min(nearest, rayBoxDistance(player, dx, dy, wall));
  return Math.min(1, nearest / 1000);
}

export class CaptureGame {
  constructor(random = rng(), mapId = MAP_SPLITS.test.start) { this.random = random; this.reset(mapId); }
  reset(mapId = this.mapId) {
    const map = generateMap(mapId);
    this.mapId = map.id; this.map = map;
    this.base = map.base; this.flag = map.flag; this.walls = map.walls;
    this.player = { x: this.base.x, y: this.base.y, vx: 0, vy: 0, health: 100, energy: 100, ammo: 30, fireCooldown: 0, dashCooldown: 0, facing: [1, 0] };
    this.enemies = map.enemies.map(([x, y], i) => ({ x, y, homeX: x, homeY: y, vx: 0, vy: 0, health: 60, cooldown: 50 + this.random() * 80, phase: this.random() * Math.PI * 2, id: i }));
    this.supplies = map.supplies.map(([x, y, type]) => ({ x, y, type, cooldown: 0 }));
    this.bullets = [];
    this.carrying = false;
    this.state = "ready";
    this.score = 0;
    this.hits = 0;
    this.kills = 0;
    this.pickups = 0;
    this.flagPickups = 0;
    this.damage = 0;
    this.frames = 0;
  }
  goal() { return this.carrying ? this.base : this.flag; }
  fire(owner, target, enemy = false) {
    const d = Math.max(1, distance(owner, target));
    this.bullets.push({ x: owner.x, y: owner.y, vx: (target.x - owner.x) / d * (enemy ? 5 : 8), vy: (target.y - owner.y) / d * (enemy ? 5 : 8), enemy, life: 180 });
  }
  step(action = null) {
    if (this.state !== "playing") return;
    this.frames++;
    const p = this.player;
    const oldX = p.x, oldY = p.y;
    p.fireCooldown = Math.max(0, p.fireCooldown - 1);
    p.dashCooldown = Math.max(0, p.dashCooldown - 1);
    p.energy = Math.min(100, p.energy + 0.08);
    const direction = action === null ? [0, 0] : DIRECTIONS[action % 8];
    const ability = action === null ? 0 : Math.floor(action / 8);
    if (direction[0] || direction[1]) p.facing = direction;
    let speed = this.carrying ? 2.4 : 3;
    if (ability === 2 && p.energy >= 20 && !p.dashCooldown) {
      p.energy -= 20;
      p.dashCooldown = 45;
      // Three collision-checked substeps avoid jumping through a wall.
      for (let i = 0; i < 3; i++) move(p, direction[0] * 5, direction[1] * 5, this.walls);
    }
    move(p, direction[0] * speed, direction[1] * speed, this.walls);
    p.vx = p.x - oldX; p.vy = p.y - oldY;
    if (ability === 1 && p.ammo > 0 && !p.fireCooldown) {
      const candidates = this.enemies.filter((e) => e.health > 0 && visible(p, e, this.walls)).sort((a, b) => distance(p, a) - distance(p, b));
      this.fire(p, candidates[0] || { x: p.x + p.facing[0], y: p.y + p.facing[1] });
      p.ammo--;
      p.fireCooldown = 14;
    }
    for (const e of this.enemies) {
      if (e.health <= 0) continue;
      e.cooldown--;
      const seesPlayer = distance(e, p) < 330 && visible(e, p, this.walls);
      const target = seesPlayer && distance(e, p) > 170 ? p : {
        x: e.homeX + Math.cos(this.frames / 130 + e.phase) * 65,
        y: e.homeY + Math.sin(this.frames / 150 + e.phase) * 65,
      };
      const d = Math.max(1, distance(e, target));
      move(e, (target.x - e.x) / d * 1.2, (target.y - e.y) / d * 1.2, this.walls);
      if (seesPlayer && e.cooldown <= 0) {
        this.fire(e, p, true);
        e.cooldown = 65 + this.random() * 35;
      }
    }
    for (const bullet of this.bullets) {
      bullet.x += bullet.vx;
      bullet.y += bullet.vy;
      bullet.life--;
      if (blocked(bullet.x, bullet.y, 2, this.walls)) { bullet.life = 0; continue; }
      if (bullet.enemy && distance(bullet, p) < RADIUS + 3) {
        const lost = Math.min(10, p.health);
        p.health -= lost;
        this.damage += lost;
        bullet.life = 0;
      } else if (!bullet.enemy) {
        const enemy = this.enemies.find((e) => e.health > 0 && distance(bullet, e) < RADIUS + 3);
        if (enemy) {
          enemy.health = Math.max(0, enemy.health - 20);
          this.hits++;
          this.score += 5;
          if (!enemy.health) { this.kills++; this.score += 15; }
          bullet.life = 0;
        }
      }
    }
    this.bullets = this.bullets.filter((b) => b.life > 0);
    if (p.health <= 0) { this.state = "lost"; return; }
    for (const supply of this.supplies) {
      supply.cooldown = Math.max(0, supply.cooldown - 1);
      const maximum = supply.type === "ammo" ? 30 : 100;
      if (!supply.cooldown && distance(p, supply) < 26 && p[supply.type] < maximum) {
        p[supply.type] = Math.min(maximum, p[supply.type] + (supply.type === "ammo" ? 20 : 40));
        supply.cooldown = 600;
        this.pickups++;
      }
    }
    if (!this.carrying && distance(p, this.flag) < 28) {
      this.carrying = true;
      this.flagPickups++;
      this.score += 25;
    }
    if (this.carrying && distance(p, this.base) < 32) { this.state = "won"; this.score += 100; }
  }
}

export function captureInputs(game) {
  const p = game.player;
  const values = [p.x / WIDTH * 2 - 1, p.y / HEIGHT * 2 - 1, p.vx / 18, p.vy / 18,
    p.health / 100, p.energy / 100, p.ammo / 30, p.fireCooldown / 14, p.dashCooldown / 45,
    Number(game.carrying), (game.flag.x - p.x) / WIDTH, (game.flag.y - p.y) / HEIGHT,
    (game.base.x - p.x) / WIDTH, (game.base.y - p.y) / HEIGHT, ...p.facing];
  const rangeInSector = (objects, direction) => {
    let nearest = 1000;
    for (const object of objects) {
      const d = distance(p, object);
      if (d < nearest && (d < 1 || ((object.x - p.x) * direction[0] + (object.y - p.y) * direction[1]) / d >= Math.cos(Math.PI / 8))) nearest = d;
    }
    return Math.min(1, nearest / 1000);
  };
  const enemies = game.enemies.filter((e) => e.health > 0 && visible(p, e, game.walls));
  const supplies = game.supplies.filter((s) => !s.cooldown && visible(p, s, game.walls));
  const bullets = game.bullets.filter((b) => b.enemy && visible(p, b, game.walls));
  for (const direction of DIRECTIONS) values.push(rayRange(p, direction, game.walls), rangeInSector(enemies, direction), rangeInSector(supplies, direction), rangeInSector(bullets, direction));
  for (const e of game.enemies) {
    const seen = e.health > 0 && visible(p, e, game.walls);
    values.push(...(seen ? [(e.x - p.x) / WIDTH, (e.y - p.y) / HEIGHT, e.vx / 1.2, e.vy / 1.2, e.health / 60, Math.min(1, distance(p, e) / 1000), 1, 1] : [0, 0, 0, 0, 0, 0, 0, Number(e.health > 0)]));
  }
  for (const s of game.supplies) values.push((s.x - p.x) / WIDTH, (s.y - p.y) / HEIGHT, ["health", "ammo", "energy"].indexOf(s.type) / 2, Number(!s.cooldown));
  return values.map((v) => clamp(v, -1, 1));
}
export function captureTransition(game, action, rewards = CTF_DEFAULT_REWARDS) {
  const objective = game.goal(), oldDistance = distance(game.player, objective);
  const before = { flags: game.flagPickups, hits: game.hits, kills: game.kills, damage: game.damage, pickups: game.pickups };
  let frames = 0;
  for (; frames < 4 && game.state === "playing"; frames++) game.step(action);
  const done = game.state === "won" || game.state === "lost";
  const terms = {
    flagPickup: (game.flagPickups - before.flags) * rewards.flagPickup,
    delivery: game.state === "won" ? rewards.delivery : 0,
    enemyHit: (game.hits - before.hits) * rewards.enemyHit,
    enemyDefeat: (game.kills - before.kills) * rewards.enemyDefeat,
    damage: (game.damage - before.damage) * rewards.damage,
    supply: (game.pickups - before.pickups) * rewards.supply,
    progress: (oldDistance - distance(game.player, objective)) / 100 * rewards.progress,
    defeat: game.state === "lost" ? rewards.defeat : 0,
    step: rewards.step,
  };
  const reward = Object.values(terms).reduce((sum, value) => sum + value, 0);
  return { reward, terms, done, frames };
}
export const CTF_ENV = {
  spec: CTF_SPEC, defaultShape: [96, 48, 32, 24],
  mapProtocol: MAP_PROTOCOL,
  createEpisode: (random, context = {}) => {
    const id = context.purpose === "evaluation"
      ? MAP_PROTOCOL.validationMapIds[(context.evaluationIndex || 0) % MAP_PROTOCOL.validationMapIds.length]
      : sampleMapId(random, "training");
    const game = new CaptureGame(random, id); game.state = "playing"; return game;
  },
  inputs: captureInputs, transition: captureTransition,
  validateRewards: (config) => validateRewards(config, CTF_REWARDS),
  score: (game) => game.score, hits: (game) => game.hits,
};

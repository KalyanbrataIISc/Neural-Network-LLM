import { rng } from "./network.js";

export const MAP_WIDTH = 960, MAP_HEIGHT = 640, CELL_SIZE = 80;
export const MAP_SPLITS = Object.freeze({
  training: Object.freeze({ start: 1, end: 1000, label: "Training" }),
  validation: Object.freeze({ start: 1001, end: 2000, label: "Validation" }),
  test: Object.freeze({ start: 2001, end: 10000, label: "Held-out test" }),
});
export const MAP_PROTOCOL = Object.freeze({
  version: "ctf-maps-v1", total: 10000, splits: MAP_SPLITS,
  validationMapIds: Object.freeze([1001, 1002, 1003, 1004, 1005]),
});
// Retained only for older physics fixtures. The map browser, training, and
// validation all use the numbered procedural library, never this old arena.
export const LEGACY_MAP = {
  id: 0, split: "legacy", seed: 0,
  base: { x: 70, y: 570 }, flag: { x: 885, y: 65 },
  walls: [{ x: 200, y: 100, w: 28, h: 210 }, { x: 200, y: 400, w: 190, h: 28 },
    { x: 450, y: 220, w: 28, h: 250 }, { x: 580, y: 140, w: 180, h: 28 }, { x: 690, y: 330, w: 28, h: 210 }],
  enemies: [[340, 130], [560, 540], [820, 260], [830, 90]],
  supplies: [[120, 340, "health"], [360, 520, "ammo"], [550, 80, "energy"], [830, 550, "health"]],
};
const cache = new Map();
const COLS = MAP_WIDTH / CELL_SIZE, ROWS = MAP_HEIGHT / CELL_SIZE;
const cellPoint = (cell) => ({ x: (cell % COLS + 0.5) * CELL_SIZE, y: (Math.floor(cell / COLS) + 0.5) * CELL_SIZE });
const edgeKey = (a, b) => `${Math.min(a, b)}:${Math.max(a, b)}`;

export function mapSplit(id) {
  if (!Number.isInteger(id) || id < 1 || id > MAP_PROTOCOL.total)
    throw new Error("Choose an integer map ID from 1 to 10,000.");
  return id <= MAP_SPLITS.training.end ? "training" : id <= MAP_SPLITS.validation.end ? "validation" : "test";
}
export function sampleMapId(random, split = "test") {
  const range = MAP_SPLITS[split];
  if (!range) throw new Error("Choose training, validation, or test maps.");
  return range.start + Math.floor(random() * (range.end - range.start + 1));
}
export function generateMap(id = MAP_SPLITS.test.start) {
  if (id === 0) return LEGACY_MAP;
  const split = mapSplit(id);
  if (cache.has(id)) return cache.get(id);
  const seed = 0x43544600 + id, random = rng(seed);
  const count = COLS * ROWS, open = new Set(), visited = new Set();
  const baseCell = Math.floor(random() * count);
  const manhattan = (a, b) => Math.abs(a % COLS - b % COLS) + Math.abs(Math.floor(a / COLS) - Math.floor(b / COLS));
  const flagCandidates = Array.from({ length: count }, (_, i) => i).filter((i) => manhattan(i, baseCell) >= 10);
  const flagCell = flagCandidates[Math.floor(random() * flagCandidates.length)];
  const neighbors = (cell) => {
    const x = cell % COLS, y = Math.floor(cell / COLS);
    return [x > 0 ? cell - 1 : -1, x < COLS - 1 ? cell + 1 : -1,
      y > 0 ? cell - COLS : -1, y < ROWS - 1 ? cell + COLS : -1].filter((n) => n >= 0);
  };
  // A spanning tree connects every room. Additional openings create loops.
  const stack = [baseCell]; visited.add(baseCell);
  while (stack.length) {
    const current = stack.at(-1), next = neighbors(current).filter((n) => !visited.has(n));
    if (!next.length) { stack.pop(); continue; }
    const chosen = next[Math.floor(random() * next.length)];
    open.add(edgeKey(current, chosen)); visited.add(chosen); stack.push(chosen);
  }
  for (let cell = 0; cell < count; cell++)
    for (const next of neighbors(cell)) if (next > cell && random() < 0.12) open.add(edgeKey(cell, next));
  // Guarantee a short, navigable objective route; no route is given to the AI.
  let current = baseCell;
  while (current !== flagCell) {
    const dx = flagCell % COLS - current % COLS;
    const dy = Math.floor(flagCell / COLS) - Math.floor(current / COLS);
    const next = dx && (!dy || random() < 0.5) ? current + Math.sign(dx) : current + Math.sign(dy) * COLS;
    open.add(edgeKey(current, next)); current = next;
  }
  const walls = [];
  for (let cell = 0; cell < count; cell++) {
    const x = cell % COLS, y = Math.floor(cell / COLS);
    if (x < COLS - 1 && !open.has(edgeKey(cell, cell + 1)))
      walls.push({ x: (x + 1) * CELL_SIZE - 4, y: y * CELL_SIZE - 4, w: 8, h: CELL_SIZE + 8 });
    if (y < ROWS - 1 && !open.has(edgeKey(cell, cell + COLS)))
      walls.push({ x: x * CELL_SIZE - 4, y: (y + 1) * CELL_SIZE - 4, w: CELL_SIZE + 8, h: 8 });
  }
  const used = new Set([baseCell, flagCell]);
  const pick = (predicate = () => true) => {
    const candidates = Array.from({ length: count }, (_, i) => i).filter((i) => !used.has(i) && predicate(i));
    const chosen = candidates[Math.floor(random() * candidates.length)];
    used.add(chosen); return cellPoint(chosen);
  };
  const enemies = Array.from({ length: 4 }, () => {
    const p = pick((cell) => manhattan(cell, baseCell) >= 4); return Object.freeze([p.x, p.y]);
  });
  const supplies = ["health", "ammo", "energy", "health"].map((type) => {
    const p = pick(); return Object.freeze([p.x, p.y, type]);
  });
  const map = Object.freeze({ id, seed, split,
    base: Object.freeze(cellPoint(baseCell)), flag: Object.freeze(cellPoint(flagCell)),
    walls: Object.freeze(walls.map(Object.freeze)),
    enemies: Object.freeze(enemies), supplies: Object.freeze(supplies),
  });
  if (cache.size >= 128) cache.delete(cache.keys().next().value);
  cache.set(id, map);
  return map;
}

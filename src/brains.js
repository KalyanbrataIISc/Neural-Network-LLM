import { Network } from "./network.js";
export class BrainStore {
  constructor(storage) {
    this.storage = storage;
    this.key = "breakout-my-brains-v1";
    this.records = [];
    const saved = storage.getItem(this.key);
    if (saved) {
      const data = JSON.parse(saved);
      if (!Array.isArray(data)) throw new Error("Invalid brain library.");
      for (const brain of data) {
        Network.load(brain.model);
        if (typeof brain.id !== "string" || typeof brain.name !== "string")
          throw new Error("Invalid brain entry.");
      }
      this.records = data;
    }
  }
  persist(next) {
    this.storage.setItem(this.key, JSON.stringify(next));
    this.records = next;
  }
  save(name, model, id = null) {
    Network.load(model);
    name = String(name).trim().slice(0, 80);
    if (!name) throw new Error("Give your brain a name.");
    const previous = this.records.find((b) => b.id === id);
    if (id && !previous) throw new Error("Brain no longer exists.");
    const entry = {
      id: previous?.id || crypto.randomUUID(),
      name,
      createdAt: previous?.createdAt || new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      model: structuredClone(model),
    };
    this.persist(
      previous
        ? this.records.map((b) => (b.id === id ? entry : b))
        : [...this.records, entry],
    );
    return entry;
  }
  rename(id, name) {
    const brain = this.get(id);
    return this.save(name, brain.model, id);
  }
  get(id) {
    const b = this.records.find((b) => b.id === id);
    if (!b) throw new Error("Brain not found.");
    return structuredClone(b);
  }
  delete(id) {
    this.get(id);
    this.persist(this.records.filter((b) => b.id !== id));
  }
}

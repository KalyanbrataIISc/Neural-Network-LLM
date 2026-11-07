import { defaultRewards, validateRewards, REWARD_LIMIT } from "./rewards.js";

// The editor owns next-run settings; loading a checkpoint does not silently
// overwrite them. A run receives a validated snapshot, never mutable UI state.
export class RewardEditor {
  constructor(root, fields, storageKey) {
    this.root = root;
    this.fields = fields;
    this.storageKey = storageKey;
    this.inputs = {};
    let initial = defaultRewards(fields);
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) initial = validateRewards(JSON.parse(saved), fields);
    } catch { /* Unavailable storage or an invalid saved preference: use defaults. */ }
    const help = document.createElement("p");
    help.textContent = "These are coefficients in the actual training reward. Zero disables a term; negative values penalize it. Edits apply to the next training run. Retrain to change behavior; saved weights do not change immediately.";
    root.append(help);
    for (const field of fields) {
      const row = document.createElement("div");
      row.className = "reward-row";
      const label = document.createElement("label");
      label.textContent = field.label;
      const input = document.createElement("input");
      input.id = `${storageKey}-${field.key}`;
      input.type = "number";
      input.step = "any";
      input.min = -REWARD_LIMIT;
      input.max = REWARD_LIMIT;
      input.value = initial[field.key];
      label.htmlFor = input.id;
      const description = document.createElement("small");
      description.id = `${input.id}-help`;
      description.textContent = field.description;
      input.setAttribute("aria-describedby", description.id);
      input.onchange = () => {
        try {
          this.save();
        } catch (error) {
          this.status.textContent = error.message;
        }
      };
      this.inputs[field.key] = input;
      row.append(label, input, description);
      root.append(row);
    }
    const limits = document.createElement("p");
    limits.textContent = "Allowed coefficients: −1,000,000 to +1,000,000. Total reward is the sum of all terms, calculated once per action (up to 4 physics frames); it is not clipped. Large coefficients can destabilize learning.";
    this.reset = document.createElement("button");
    this.reset.className = "text-button";
    this.reset.textContent = "↻ Restore default rewards";
    this.reset.onclick = () => {
      for (const field of fields) this.inputs[field.key].value = field.value;
      this.save();
    };
    this.status = document.createElement("p");
    this.status.setAttribute("role", "status");
    this.status.className = "reward-status";
    this.status.textContent = "Ready for the next run. Settings save in this browser.";
    root.append(limits, this.reset, this.status);
  }
  read() {
    const config = {};
    for (const { key } of this.fields) {
      const input = this.inputs[key];
      config[key] = input.value === "" ? NaN : Number(input.value);
      input.setAttribute("aria-invalid", String(!input.validity.valid || input.value === ""));
    }
    return validateRewards(config, this.fields);
  }
  save() {
    const config = this.read();
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(config));
      this.status.textContent = "Saved for the next run. Current weights are unchanged.";
    } catch {
      this.status.textContent = "Ready for the next run; browser storage is unavailable.";
    }
    return config;
  }
  lock(locked) {
    for (const input of Object.values(this.inputs)) input.disabled = locked;
    this.reset.disabled = locked;
    this.status.textContent = locked
      ? "Training uses the values shown. Stop training to edit rewards."
      : "Ready for the next run. Retrain to try the effect of your rewards.";
  }
  describe(config) {
    if (!config) return "Exact reward coefficients were not recorded in this checkpoint. Next-run settings are shown in the reward editor.";
    try {
      const values = validateRewards(config, this.fields);
      return this.fields.map(({ key, label }) => `${label}: ${values[key]}`).join(" · ");
    } catch { return "This checkpoint has unrecognized reward metadata."; }
  }
}

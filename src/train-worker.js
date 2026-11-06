import { trainingSession } from "./trainer.js";
let iterator = null,
  mode = "automatic",
  scheduled = false,
  last = null,
  total = Infinity;
function queue() {
  if (!scheduled) {
    scheduled = true;
    setTimeout(() => {
      scheduled = false;
      if (mode === "automatic") advance();
    }, 0);
  }
}
function advance() {
  if (!iterator) return;
  try {
    const next = iterator.next();
    if (next.done) {
      self.postMessage({ type: "complete", model: next.value });
      iterator = null;
      return;
    }
    last = next.value;
    if (mode === "manual") self.postMessage({ type: "manual-step", ...last });
    else queue();
    if (last.episode >= total) {
      const final = iterator.next();
      self.postMessage({
        type: "complete",
        model: mode === "manual" ? last.model : final.value,
      });
      iterator = null;
    }
  } catch (error) {
    iterator = null;
    self.postMessage({ type: "error", message: error.message });
  }
}
self.onmessage = ({ data }) => {
  if (data.command === "pause") {
    mode = "manual";
    if (last) self.postMessage({ type: "manual-paused", ...last });
    return;
  }
  mode = data.command === "step" ? "manual" : "automatic";
  if (!iterator)
    iterator = trainingSession({
      episodes: data.episodes,
      initialModel: data.model,
      resetWeights: data.resetWeights,
      plan: data.plan,
      order: data.order,
      gameId: data.gameId,
      isManual: () => mode === "manual",
      onCheckpoint: (model) => {
        if (mode === "automatic")
          self.postMessage({ type: "checkpoint", model });
      },
      onProgress: (progress) => {
        total = progress.episodes;
        self.postMessage({ type: "progress", ...progress });
      },
    });
  // A queued automatic callback becomes a single manual step if paused meanwhile.
  if (!scheduled) advance();
};

import { train } from "./trainer.js";
self.onmessage = ({ data }) => {
  try {
    const model = train({
      episodes: data.episodes,
      initialModel: data.model,
      resetWeights: data.resetWeights,
      onCheckpoint: (model) => self.postMessage({ type: "checkpoint", model }),
      onProgress: (progress) =>
        self.postMessage({ type: "progress", ...progress }),
    });
    self.postMessage({ type: "complete", model });
  } catch (error) {
    self.postMessage({ type: "error", message: error.message });
  }
};

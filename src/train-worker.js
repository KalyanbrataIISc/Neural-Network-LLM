import { train } from "./trainer.js";
import { CTF_ENV } from "./ctf-game.js";
self.onmessage = ({ data }) => {
  try {
    const model = train({
      episodes: data.episodes,
      initialModel: data.model,
      resetWeights: data.resetWeights,
      rewardConfig: data.rewardConfig,
      checkpointSelection: data.checkpointSelection,
      ...(data.game === "ctf" ? { environment: CTF_ENV } : {}),
      onCheckpoint: (model) => self.postMessage({ type: "checkpoint", model }),
      onProgress: (progress) =>
        self.postMessage({ type: "progress", ...progress }),
    });
    self.postMessage({ type: "complete", model });
  } catch (error) {
    self.postMessage({ type: "error", message: error.message });
  }
};

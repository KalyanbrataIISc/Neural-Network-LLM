import { train, evaluate } from "../src/trainer.js";
import { Network } from "../src/network.js";
import { writeFileSync, readFileSync, mkdirSync, renameSync } from "node:fs";
import { dirname } from "node:path";
const args = process.argv.slice(2),
  option = (name, fallback) =>
    args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const episodes = Number(option("--episodes", "5000")),
  output = option("--out", "models/pretrained.json"),
  resume = option("--resume", null);
const initialModel = resume ? JSON.parse(readFileSync(resume, "utf8")) : null;
mkdirSync(dirname(output), { recursive: true });
function save(model) {
  writeFileSync(output + ".tmp", JSON.stringify(model, null, 2) + "\n");
  renameSync(output + ".tmp", output);
}
console.log(
  `Training Double DQN for ${episodes.toLocaleString()} episodes; saving best weights to ${output}`,
);
const model = train({
  episodes,
  initialModel,
  onCheckpoint: save,
  onProgress: (p) => {
    if (p.episode % 100 === 0 || p.episode === episodes)
      console.log(
        `Episode ${p.episode}/${p.episodes} | eval score ${p.score.toFixed(0)} | hits ${p.hits.toFixed(1)} | ${p.frames.toLocaleString()} frames | ${p.framesPerSecond.toLocaleString()} frames/s | exploration ${(p.epsilon * 100).toFixed(1)}%`,
      );
  },
});
save(model);
console.log("Best checkpoint evaluation:", evaluate(Network.load(model)));

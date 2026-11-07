import { train, evaluate } from "../src/trainer.js";
import { CTF_ENV, CTF_SPEC } from "../src/ctf-game.js";
import { Network } from "../src/network.js";
import { writeFileSync, readFileSync, mkdirSync, renameSync } from "node:fs";
import { dirname } from "node:path";
const args = process.argv.slice(2),
  option = (name, fallback) =>
    args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const gameName = option("--game", "breakout");
if (!["breakout", "ctf"].includes(gameName)) throw new Error("Use --game breakout or --game ctf.");
const environment = gameName === "ctf" ? CTF_ENV : undefined;
const episodes = Number(option("--episodes", "5000")),
  output = option("--out", gameName === "ctf" ? "models/robot-capture.json" : "models/pretrained.json"),
  resume = option("--resume", null),
  rewardsFile = option("--rewards", null);
const rewardConfig = rewardsFile ? JSON.parse(readFileSync(rewardsFile, "utf8")) : undefined;
const initialModel = resume ? JSON.parse(readFileSync(resume, "utf8")) : null;
mkdirSync(dirname(output), { recursive: true });
function save(model) {
  writeFileSync(output + ".tmp", JSON.stringify(model, null, 2) + "\n");
  renameSync(output + ".tmp", output);
}
console.log(
  `Training ${gameName} Double DQN for ${episodes.toLocaleString()} episodes; saving checkpoints to ${output}`,
);
const model = train({
  episodes,
  initialModel,
  rewardConfig,
  checkpointSelection: option("--save-policy", "score"),
  environment,
  onCheckpoint: save,
  onProgress: (p) => {
    if (p.episode % 100 === 0 || p.episode === episodes)
      console.log(
        `Episode ${p.episode}/${p.episodes} | eval score ${p.score.toFixed(0)} | hits ${p.hits.toFixed(1)} | ${p.frames.toLocaleString()} frames | ${p.framesPerSecond.toLocaleString()} frames/s | exploration ${(p.epsilon * 100).toFixed(1)}%`,
      );
  },
});
save(model);
console.log("Run reward coefficients:", model.rewardConfig);
console.log("Saved checkpoint evaluation:", evaluate(Network.load(model, gameName === "ctf" ? CTF_SPEC : undefined), 5, 3600, environment));

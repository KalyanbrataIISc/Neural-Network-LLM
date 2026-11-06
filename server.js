import http from "node:http";
import { readFile } from "node:fs/promises";
import { resolve, extname } from "node:path";
const root = resolve(import.meta.dirname);
const types = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
};
http
  .createServer(async (req, res) => {
    try {
      if (req.url.startsWith("/api/native/")) {
        const origin = req.headers.origin;
        if (origin && origin !== `http://${req.headers.host}`) {
          res.writeHead(403).end();
          return;
        }
        const suffix = req.url.slice("/api/native".length);
        if (!/^\/(status|jobs(?:\/[a-f0-9]{32})?)$/.test(suffix)) {
          res.writeHead(404).end();
          return;
        }
        try {
          let body = "";
          for await (const chunk of req) {
            body += chunk;
            if (body.length > 1500000) throw new Error("Request too large.");
          }
          const response = await fetch("http://127.0.0.1:3001" + suffix, {
            method: req.method,
            headers: { "Content-Type": "application/json" },
            body: ["GET", "HEAD"].includes(req.method) ? undefined : body,
            signal: AbortSignal.timeout(15000),
          });
          res
            .writeHead(response.status, {
              "Content-Type": "application/json",
              "Cache-Control": "no-store",
            })
            .end(await response.text());
        } catch {
          res.writeHead(503, { "Content-Type": "application/json" }).end(
            JSON.stringify({
              error:
                "Native service unavailable. Install backend/requirements.txt and run npm run gpu in another terminal.",
            }),
          );
        }
        return;
      }
      const path = resolve(
        root,
        "." +
          decodeURIComponent(
            new URL(req.url, "http://localhost").pathname === "/"
              ? "/index.html"
              : new URL(req.url, "http://localhost").pathname,
          ),
      );
      if (!path.startsWith(root + "/") || path.includes("/.")) {
        res.writeHead(403).end();
        return;
      }
      const data = await readFile(path);
      res
        .writeHead(200, {
          "Content-Type": types[extname(path)] || "application/octet-stream",
          "Cache-Control": "no-cache",
        })
        .end(data);
    } catch {
      res.writeHead(404).end("Not found");
    }
  })
  .listen(Number(process.env.PORT || 3000), "0.0.0.0", () =>
    console.log("Breakout Lab: http://localhost:" + (process.env.PORT || 3000)),
  );

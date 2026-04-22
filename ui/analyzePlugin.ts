import type { Plugin } from "vite";
import { spawn } from "child_process";
import { createReadStream, createWriteStream, existsSync, mkdirSync, statSync } from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";
import { fileURLToPath } from "url";
import Busboy from "busboy";
import { lookup } from "mime-types";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

// Absolute paths
const PROJECT_ROOT = resolve(__dirname, "..");
const OUTPUT_DIR   = join(PROJECT_ROOT, "output");
const SCRIPT_PATH  = join(PROJECT_ROOT, "run_classify.py");
const PYTHON       = "/Users/alidaeihagh/anaconda3/envs/ski-pose-estimation/bin/python";

export function analyzePlugin(): Plugin {
    return {
        name: "ski-analyze",
        configureServer(server) {

            // Serve output/ at /output/* so the browser can stream the annotated video
            server.middlewares.use("/output", (req, res, next) => {
                const filePath = join(OUTPUT_DIR, req.url ?? "");
                if (existsSync(filePath) && statSync(filePath).isFile()) {
                    const stat = statSync(filePath);
                    res.setHeader("Content-Length", stat.size);
                    res.setHeader("Content-Type", lookup(filePath) || "application/octet-stream");
                    res.setHeader("Accept-Ranges", "bytes");
                    createReadStream(filePath).pipe(res);
                } else {
                    next();
                }
            });

            // Handle POST /api/analyze
            server.middlewares.use("/api/analyze", (req, res, next) => {
                if (req.method !== "POST") return next();

                const busboy = Busboy({ headers: req.headers });
                let tmpVideoPath: string | null = null;
                let videoStem = `video-${Date.now()}`;

                busboy.on("file", (_field, stream, info) => {
                    const ext = info.filename.split(".").pop() ?? "mp4";
                    videoStem = info.filename.replace(/\.[^.]+$/, "");
                    tmpVideoPath = join(tmpdir(), `ski-${Date.now()}.${ext}`);
                    stream.pipe(createWriteStream(tmpVideoPath));
                });

                busboy.on("finish", () => {
                    if (!tmpVideoPath) {
                        res.statusCode = 400;
                        res.end(JSON.stringify({ error: "No video file uploaded." }));
                        return;
                    }

                    if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });
                    const outputPath = join(OUTPUT_DIR, `classified_${videoStem}.mp4`);

                    console.log(`[analyze] Running classification on ${tmpVideoPath}`);

                    const py = spawn(PYTHON, [
                        SCRIPT_PATH,
                        "--input", tmpVideoPath,
                        "--output-path", outputPath,
                    ], { cwd: PROJECT_ROOT });

                    let stdout = "";
                    let stderr = "";
                    py.stdout.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
                    py.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); process.stderr.write(chunk); });

                    py.on("close", (code) => {
                        if (code !== 0) {
                            console.error(`[analyze] Python exited with code ${code}`);
                            res.statusCode = 500;
                            res.end(JSON.stringify({ error: stderr.slice(-500) || "Classification failed." }));
                            return;
                        }

                        // ML libs print noise before the JSON — take the last JSON line
                        const jsonLine = stdout.trim().split("\n").reverse().find(l => l.trimStart().startsWith("{"));
                        if (!jsonLine) {
                            res.statusCode = 500;
                            res.end(JSON.stringify({ error: "No JSON output from classifier.", detail: stdout.slice(-500) }));
                            return;
                        }

                        try {
                            const data = JSON.parse(jsonLine);
                            data.annotated_video_url = `/output/classified_${videoStem}.mp4`;
                            res.setHeader("Content-Type", "application/json");
                            res.end(JSON.stringify(data));
                            console.log(`[analyze] Done — ${data.level} (${(data.confidence * 100).toFixed(1)}%)`);
                        } catch {
                            res.statusCode = 500;
                            res.end(JSON.stringify({ error: "Failed to parse classifier output." }));
                        }
                    });
                });

                req.pipe(busboy);
            });
        },
    };
}

import "dotenv/config";
import cors from "cors";
import express from "express";
import multer from "multer";
import sharp from "sharp";
import * as ort from "onnxruntime-node";
import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { fileURLToPath } from "node:url";
import { MongoClient } from "mongodb";
import { CLASSES, type CurrencyCode } from "./src/banknotes.ts";

const app = express();
const port = Number(process.env.PORT ?? 8787);
const modelPath = fileURLToPath(new URL("./public/model.onnx", import.meta.url));
const inputSize = 640;
const confidenceFloor = 0.35;
const elevenLabsApi = "https://api.elevenlabs.io/v1";
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 12 * 1024 * 1024 } });
const mongoClient = new MongoClient(process.env.MONGODB_URI ?? "mongodb://127.0.0.1:27017");
const users = mongoClient.db(process.env.MONGODB_DB ?? "currency_vision").collection("users");
const history = mongoClient.db(process.env.MONGODB_DB ?? "currency_vision").collection("history");

app.use(cors());
app.use(express.json({ limit: "2mb" }));
app.use((req, res, next) => {
  const startedAt = Date.now();
  res.on("finish", () => {
    console.log("[request]", {
      method: req.method,
      path: req.path,
      status: res.statusCode,
      durationMs: Date.now() - startedAt,
    });
  });
  next();
});
// Reject malformed JSON requests cleanly. This keeps accidental probes or
// browser-extension traffic from printing an Express parser stack trace.
app.use((error: unknown, _req: express.Request, res: express.Response, next: express.NextFunction) => {
  if (error instanceof SyntaxError && "body" in error) {
    res.status(400).json({ error: "Invalid JSON request body" });
    return;
  }
  next(error);
});
app.get("/health", (_req, res) => res.json({ ok: true, model: "currency-yolo" }));

app.post("/api/auth/signup", async (req, res) => {
  const { name, email, phone, password, language, conversionCurrency } = req.body as Record<string, string>;
  const normalizedEmail = email?.trim().toLowerCase();
  if (!name?.trim() || !normalizedEmail || !phone?.trim() || !password || password.length < 6) {
    res.status(400).json({ error: "Please provide valid signup details." });
    return;
  }

  try {
    const existing = await users.findOne({ email: normalizedEmail });
    if (existing) {
      res.status(409).json({ error: "An account with this email already exists." });
      return;
    }
    const salt = randomBytes(16).toString("hex");
    const passwordHash = scryptSync(password, salt, 64).toString("hex");
    await users.insertOne({
      name: name.trim(),
      email: normalizedEmail,
      phone: phone.trim(),
      passwordHash: `${salt}:${passwordHash}`,
      language: language || "kn-IN",
      conversionCurrency: conversionCurrency || "USD",
      createdAt: new Date(),
    });
    console.log("[auth] signup success", {
      email: maskEmail(normalizedEmail),
      language: language || "kn-IN",
      conversionCurrency: conversionCurrency || "USD",
    });
    res.status(201).json({ name: name.trim(), email: normalizedEmail, phone: phone.trim(), language: language || "kn-IN", conversionCurrency: conversionCurrency || "USD" });
  } catch (error) {
    console.error("Signup failed:", error instanceof Error ? error.message : "unknown error");
    res.status(503).json({ error: "Local MongoDB is unavailable." });
  }
});

app.post("/api/auth/login", async (req, res) => {
  const { email, password } = req.body as Record<string, string>;
  const normalizedEmail = email?.trim().toLowerCase();
  if (!normalizedEmail || !password) {
    res.status(400).json({ error: "Email and password are required." });
    return;
  }

  try {
    const user = await users.findOne<{ name: string; email: string; phone?: string; passwordHash: string; language?: string; conversionCurrency?: string }>({ email: normalizedEmail });
    if (!user) {
      console.log("[auth] login failed", { email: maskEmail(normalizedEmail), reason: "user not found" });
      res.status(401).json({ error: "Invalid email or password." });
      return;
    }
    const [salt, storedHash] = user.passwordHash.split(":");
    const suppliedHash = scryptSync(password, salt, 64);
    const valid = storedHash && timingSafeEqual(suppliedHash, Buffer.from(storedHash, "hex"));
    if (!valid) {
      console.log("[auth] login failed", { email: maskEmail(normalizedEmail), reason: "invalid password" });
      res.status(401).json({ error: "Invalid email or password." });
      return;
    }
    console.log("[auth] login success", {
      email: maskEmail(normalizedEmail),
      language: user.language || "kn-IN",
      conversionCurrency: user.conversionCurrency || "USD",
    });
    res.json({ name: user.name, email: user.email, phone: user.phone, language: user.language, conversionCurrency: user.conversionCurrency });
  } catch (error) {
    console.error("Login failed:", error instanceof Error ? error.message : "unknown error");
    res.status(503).json({ error: "Local MongoDB is unavailable." });
  }
});

app.get("/api/auth/profile", async (req, res) => {
  const email = String(req.query.email ?? "").trim().toLowerCase();
  if (!email) {
    res.status(400).json({ error: "Email is required." });
    return;
  }
  try {
    const user = await users.findOne(
      { email },
      { projection: { _id: 0, name: 1, email: 1, phone: 1, language: 1, conversionCurrency: 1, createdAt: 1 } },
    );
    if (!user) {
      res.status(404).json({ error: "Profile not found." });
      return;
    }
    console.log("[profile] loaded", {
      email: maskEmail(email),
      language: user.language || "kn-IN",
      conversionCurrency: user.conversionCurrency || "USD",
    });
    res.json(user);
  } catch (error) {
    console.error("Profile lookup failed:", error instanceof Error ? error.message : "unknown error");
    res.status(503).json({ error: "Local MongoDB is unavailable." });
  }
});

app.get("/api/history", async (req, res) => {
  const email = String(req.query.email ?? "").trim().toLowerCase();
  if (!email) {
    res.status(400).json({ error: "Email is required." });
    return;
  }
  try {
    const records = await history.find({ email }, { projection: { _id: 0, record: 1 } }).sort({ "record.createdAt": -1 }).toArray();
    console.log("[history] loaded", { email: maskEmail(email), count: records.length });
    res.json(records.map((item) => item.record));
  } catch (error) {
    console.error("History lookup failed:", error instanceof Error ? error.message : "unknown error");
    res.status(503).json({ error: "Local MongoDB is unavailable." });
  }
});

app.post("/api/history", async (req, res) => {
  const { email, record } = req.body as { email?: string; record?: Record<string, unknown> };
  const normalizedEmail = email?.trim().toLowerCase();
  if (!normalizedEmail || !record?.id) {
    res.status(400).json({ error: "Email and history record are required." });
    return;
  }
  try {
    await history.updateOne(
      { email: normalizedEmail, "record.id": record.id },
      { $set: { email: normalizedEmail, record } },
      { upsert: true },
    );
    console.log("[history] saved", { email: maskEmail(normalizedEmail), recordId: record.id });
    res.status(201).json(record);
  } catch (error) {
    console.error("History save failed:", error instanceof Error ? error.message : "unknown error");
    res.status(503).json({ error: "Local MongoDB is unavailable." });
  }
});

app.delete("/api/history/:id", async (req, res) => {
  const email = String(req.query.email ?? "").trim().toLowerCase();
  if (!email) {
    res.status(400).json({ error: "Email is required." });
    return;
  }
  try {
    await history.deleteOne({ email, "record.id": req.params.id });
    console.log("[history] deleted", { email: maskEmail(email), recordId: req.params.id });
    res.status(204).end();
  } catch (error) {
    console.error("History delete failed:", error instanceof Error ? error.message : "unknown error");
    res.status(503).json({ error: "Local MongoDB is unavailable." });
  }
});

app.delete("/api/history", async (req, res) => {
  const email = String(req.query.email ?? "").trim().toLowerCase();
  if (!email) {
    res.status(400).json({ error: "Email is required." });
    return;
  }
  try {
    await history.deleteMany({ email });
    console.log("[history] cleared", { email: maskEmail(email) });
    res.status(204).end();
  } catch (error) {
    console.error("History clear failed:", error instanceof Error ? error.message : "unknown error");
    res.status(503).json({ error: "Local MongoDB is unavailable." });
  }
});

app.post("/api/tts", upload.single("audio"), async (req, res) => {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    console.error("[tts] unavailable: ELEVENLABS_API_KEY is not configured");
    res.status(503).json({ error: "ELEVENLABS_API_KEY is not configured" });
    return;
  }

  try {
    if (req.body.action === "voices") {
      console.log("[tts] voices requested", { language: req.body.languageCode || "all" });
      const response = await fetch(`${elevenLabsApi}/voices`, { headers: { "xi-api-key": apiKey } });
      const data = await response.json();
      res.status(response.status).json(data);
      return;
    }

    if (req.body.action === "transcribe") {
      console.log("[tts] transcription requested", { language: req.body.languageCode || "auto" });
      if (!req.file) {
        res.status(400).json({ error: "audio is required" });
        return;
      }
      const form = new FormData();
      form.append("model_id", "scribe_v1");
      form.append("file", new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname || "voice-command.m4a");
      const language = String(req.body.languageCode || "").split("-")[0];
      if (language) form.append("language_code", language);
      const response = await fetch(`${elevenLabsApi}/speech-to-text`, { method: "POST", headers: { "xi-api-key": apiKey }, body: form });
      const data = await response.json();
      res.status(response.status).json(data);
      return;
    }

    if (req.body.action === "synthesize") {
      const voiceId = String(req.body.voiceId || "");
      const text = String(req.body.text || "");
      if (!voiceId || !text) {
        res.status(400).json({ error: "voiceId and text are required" });
        return;
      }
      console.log("[tts] synthesis requested", {
        language: req.body.languageCode || "auto",
        voiceId,
        textLength: text.length,
      });
      const response = await fetchElevenLabs(`${elevenLabsApi}/text-to-speech/${voiceId}?output_format=mp3_44100_128`, {
        method: "POST",
        headers: { "xi-api-key": apiKey, "Content-Type": "application/json", Accept: "audio/mpeg" },
        body: JSON.stringify({
          text,
          language_code: String(req.body.languageCode || "").split("-")[0] || undefined,
          model_id: "eleven_v3",
          voice_settings: { stability: 0.5, similarity_boost: 0.75, style: 0.4, use_speaker_boost: true, speed: Number(req.body.speed) || 1 },
        }),
      });
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          res.status(502).json({ error: "ElevenLabs API key was rejected. Update ELEVENLABS_API_KEY." });
          return;
        }
        if (response.status === 429) {
          res.status(429).json({ error: "ElevenLabs rate limit reached. Please wait a moment and try again." });
          return;
        }
        res.status(response.status).json({ error: "ElevenLabs synthesis failed", detail: await response.text() });
        return;
      }
      const audio = Buffer.from(await response.arrayBuffer()).toString("base64");
      console.log("[tts] synthesis success", {
        language: req.body.languageCode || "auto",
        voiceId,
        audioBytes: audio.length,
      });
      res.json({ audio, contentType: "audio/mpeg", duration: 0 });
      return;
    }

    res.status(400).json({ error: "Unsupported TTS action" });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown error";
    console.error("TTS request failed:", detail);
    res.status(502).json({ error: "TTS provider request failed", detail });
  }
});

async function fetchElevenLabs(url: string, init: RequestInit): Promise<Response> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      return await fetch(url, { ...init, signal: AbortSignal.timeout(30_000) });
    } catch (error) {
      lastError = error;
      if (attempt === 0) await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("ElevenLabs connection failed");
}

let sessionPromise: Promise<ort.InferenceSession> | undefined;
function loadSession() {
  sessionPromise ??= ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });
  return sessionPromise;
}

const modelWarmupPromise = (async () => {
  const startedAt = Date.now();
  const session = await loadSession();
  const inputName = session.inputNames[0];
  const warmupData = new Float32Array(3 * inputSize * inputSize);
  await session.run({
    [inputName]: new ort.Tensor("float32", warmupData, [1, 3, inputSize, inputSize]),
  });
  console.log("[model] ready", { warmupMs: Date.now() - startedAt, inputSize });
})();

app.post("/api/inference", express.raw({ type: ["image/jpeg", "image/png", "application/octet-stream"], limit: "12mb" }), async (req, res) => {
  try {
    await modelWarmupPromise;
    if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
      res.status(400).json({ error: "An image body is required" });
      return;
    }

    const currency = req.header("x-detection-currency") as CurrencyCode | undefined;
    const currencies = new Set(
      (req.header("x-detection-currencies") ?? currency ?? "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    );
    const mode = req.header("x-detection-mode") ?? "single";
    const threshold = Number(req.header("x-confidence-threshold")) || 0.65;
    const inferenceStartedAt = Date.now();
    const { data, info } = await sharp(req.body)
      .resize({ width: inputSize, height: inputSize, fit: "contain", background: { r: 114, g: 114, b: 114 } })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    const preprocessingMs = Date.now() - inferenceStartedAt;

    const planeSize = info.width * info.height;
    const tensorData = new Float32Array(planeSize * 3);
    for (let index = 0; index < planeSize; index += 1) {
      tensorData[index] = data[index * 3] / 255;
      tensorData[planeSize + index] = data[index * 3 + 1] / 255;
      tensorData[planeSize * 2 + index] = data[index * 3 + 2] / 255;
    }

    const session = await loadSession();
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const output = (await session.run({
      [inputName]: new ort.Tensor("float32", tensorData, [1, 3, info.height, info.width]),
    }))[outputName];
    const modelInferenceMs = Date.now() - inferenceStartedAt - preprocessingMs;

    const [, boxCount, columns] = output.dims as [number, number, number];
    const values = output.data as Float32Array;
    const rawPredictions: Array<{
      index: number;
      confidence: number;
      classIndex: number;
      detectedClass: string | null;
      accepted: boolean;
      reason?: string;
    }> = [];
    const detections: Array<{ currency: CurrencyCode; denomination: number; confidence: number }> = [];
    for (let index = 0; index < boxCount; index += 1) {
      const base = index * columns;
      const confidence = values[base + 4];
      const classIndex = Math.round(values[base + 5]);
      const note = CLASSES[classIndex];
      let reason: string | undefined;
      // Keep usable low-confidence detections so the frontend can route them
      // to the counterfeit/security check instead of treating them as absent.
      if (confidence < confidenceFloor) reason = "below confidence floor";
      else if (!note) reason = "unknown class";
      else if (currencies.size > 0 && !currencies.has(note.currency)) reason = "currency filtered";
      if (confidence >= confidenceFloor || reason === "currency filtered") {
        rawPredictions.push({
          index,
          confidence,
          classIndex,
          detectedClass: note ? `${note.currency} ${note.value}` : null,
          accepted: !reason,
          ...(reason ? { reason } : {}),
        });
      }
      if (reason) continue;
      detections.push({ currency: note.currency as CurrencyCode, denomination: note.value, confidence });
    }

    const selected = mode === "multiple" ? detections : detections.slice(0, 1);
    const resultConfidence = selected[0]?.confidence ?? 0;
    const result = {
      status: selected.length === 0 ? "error" : resultConfidence < threshold ? "low_confidence" : "success",
      currencies: selected,
      total: selected.reduce((sum, note) => sum + note.denomination, 0),
      possibleReasons: resultConfidence < threshold ? ["Poor lighting", "Note at a sharp angle", "Partial or folded note"] : undefined,
      elapsedMs: undefined,
    };
    console.log("[inference] prediction", {
      currency,
      currencies: [...currencies],
      mode,
      threshold,
      inputBytes: req.body.length,
      preprocessingMs,
      modelInferenceMs,
      totalInferenceMs: Date.now() - inferenceStartedAt,
      inputDimensions: { width: info.width, height: info.height, channels: info.channels },
      model: {
        inputName,
        outputName,
        outputDimensions: output.dims,
        boxCount,
        columns,
      },
      rawPredictions,
      acceptedDetections: detections,
      output: result,
    });
    res.json(result);
  } catch (error) {
    console.error("[inference] failed", error instanceof Error ? error.message : "unknown error");
    res.status(500).json({ error: "Model inference failed" });
  }
});

app.listen(port, () => console.log(`Currency model API listening on http://localhost:${port}`));

function maskEmail(email: string): string {
  const [local, domain] = email.split("@");
  if (!domain) return "***";
  return `${local.slice(0, 2)}***@${domain}`;
}

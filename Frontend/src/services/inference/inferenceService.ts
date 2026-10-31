import type { DetectionResult, UserPreferences } from "@/types";
import { apiUrl } from "@/services/api";
import * as ort from "onnxruntime-web";

export interface InferenceOutput {
  result: DetectionResult;
}

export interface FrameQuality {
  ready: boolean;
  message: string;
}

export interface CapturedFrame {
  blob: Blob;
  width: number;
  height: number;
}

interface CurrencyValidationResult {
  valid: boolean;
}

const FRONTEND_MODEL_URL = "/model.onnx";
const FRONTEND_MODEL_CONFIDENCE_FLOOR = 0.35;
const MODEL_CLASSES = [
  ["USD", 50], ["USD", 5], ["USD", 1], ["USD", 10], ["USD", 20], ["USD", 100],
  ["PHP", 100], ["PHP", 1000], ["PHP", 1000], ["PHP", 20], ["PHP", 200], ["PHP", 50], ["PHP", 500],
  ["INR", 100], ["INR", 10], ["INR", 200], ["INR", 20], ["INR", 500], ["INR", 50], ["INR", 2000],
  ["EUR", 100], ["EUR", 10], ["EUR", 200], ["EUR", 20], ["EUR", 50], ["EUR", 5],
  ["AUD", 100], ["AUD", 10], ["AUD", 20], ["AUD", 50], ["AUD", 5],
  ["CAD", 100], ["CAD", 10], ["CAD", 20], ["CAD", 50], ["CAD", 5],
] as const;

let frontendSessionPromise: Promise<ort.InferenceSession> | undefined;

function loadFrontendModel(): Promise<ort.InferenceSession> {
  frontendSessionPromise ??= ort.InferenceSession.create(FRONTEND_MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return frontendSessionPromise;
}

export async function checkFrameQuality(video: HTMLVideoElement): Promise<FrameQuality> {
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    return { ready: false, message: "Camera is still starting. Please wait a moment." };
  }

  const first = captureFrame(video);
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  const second = captureFrame(video);
  if (!first || !second) {
    return { ready: false, message: "Please place the currency note clearly inside the frame." };
  }

  const brightness = averageBrightness(second);
  if (brightness < 28) {
    return { ready: false, message: "The image is too dark. Improve the lighting and show the note clearly." };
  }
  if (brightness > 235) {
    return { ready: false, message: "The image is too bright. Reduce glare and show the note clearly." };
  }
  if (frameDifference(first, second) > 24) {
    return { ready: false, message: "Hold the phone steady with the full note inside the frame." };
  }
  if (sharpness(second) < 14) {
    return { ready: false, message: "The note is not clear. Move closer and hold the phone steady." };
  }

  return { ready: true, message: "Clear frame captured. Checking the note…" };
}

export async function captureCameraFrame(
  video: HTMLVideoElement,
  captureRegion?: HTMLElement,
): Promise<CapturedFrame> {
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    throw new Error("Camera frame is not ready. Please wait for the camera preview and try again.");
  }

  const videoRect = video.getBoundingClientRect();
  const regionRect = captureRegion?.getBoundingClientRect() ?? videoRect;
  const scale = Math.max(
    videoRect.width / video.videoWidth,
    videoRect.height / video.videoHeight,
  );
  const renderedWidth = video.videoWidth * scale;
  const renderedHeight = video.videoHeight * scale;
  const offsetX = (videoRect.width - renderedWidth) / 2;
  const offsetY = (videoRect.height - renderedHeight) / 2;
  const paddingX = regionRect.width * 0.1;
  const paddingY = regionRect.height * 0.1;
  const sourceX = Math.max(
    0,
    (regionRect.left - videoRect.left - offsetX - paddingX) / scale,
  );
  const sourceY = Math.max(
    0,
    (regionRect.top - videoRect.top - offsetY - paddingY) / scale,
  );
  const sourceWidth = Math.min(
    video.videoWidth - sourceX,
    (regionRect.width + paddingX * 2) / scale,
  );
  const sourceHeight = Math.min(
    video.videoHeight - sourceY,
    (regionRect.height + paddingY * 2) / scale,
  );
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(sourceWidth));
  canvas.height = Math.max(1, Math.round(sourceHeight));
  canvas.getContext("2d")?.drawImage(
    video,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    canvas.width,
    canvas.height,
  );
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", 0.9),
  );
  if (!blob) throw new Error("Could not capture a camera frame");
  return { blob, width: canvas.width, height: canvas.height };
}

export async function validateCurrencyFrame(blob: Blob): Promise<CurrencyValidationResult> {
  const imageBitmap = await createImageBitmap(blob);
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 640;
  canvas.getContext("2d")?.drawImage(imageBitmap, 0, 0, 640, 640);
  imageBitmap.close();
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Could not prepare the captured frame for validation");
  const { data } = context.getImageData(0, 0, 640, 640);
  const planeSize = 640 * 640;
  const tensorData = new Float32Array(planeSize * 3);
  for (let index = 0; index < planeSize; index += 1) {
    tensorData[index] = data[index * 4] / 255;
    tensorData[planeSize + index] = data[index * 4 + 1] / 255;
    tensorData[planeSize * 2 + index] = data[index * 4 + 2] / 255;
  }
  const session = await loadFrontendModel();
  const outputName = session.outputNames[0];
  const output = (await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", tensorData, [1, 3, 640, 640]),
  }))[outputName];
  const [, boxCount, columns] = output.dims as [number, number, number];
  const values = output.data as Float32Array;
  for (let index = 0; index < boxCount; index += 1) {
    const base = index * columns;
    const confidence = values[base + 4];
    const classIndex = Math.round(values[base + 5]);
    if (confidence >= FRONTEND_MODEL_CONFIDENCE_FLOOR && MODEL_CLASSES[classIndex]) {
      return { valid: true };
    }
  }
  return { valid: false };
}

/** Sends the already-captured camera frame to the ONNX inference API. */
export async function runInference(
  preferences: UserPreferences,
  capturedFrame: CapturedFrame,
): Promise<InferenceOutput> {
  const response = await fetch(apiUrl("/api/inference"), {
    method: "POST",
    headers: {
      "Content-Type": "image/jpeg",
      "X-Detection-Currency": preferences.detectionCurrency,
      "X-Detection-Currencies": preferences.detectionMode === "multiple"
        ? preferences.selectedCurrencies.join(",")
        : preferences.detectionMode === "automatic"
        ? "INR,USD,PHP,EUR,AUD,CAD"
        : preferences.detectionCurrency,
      "X-Detection-Mode": preferences.detectionMode,
      "X-Confidence-Threshold": String(preferences.confidenceThreshold),
    },
    body: capturedFrame.blob,
  });
  if (!response.ok) throw new Error(`Model API request failed (${response.status})`);
  return { result: (await response.json()) as DetectionResult };
}

function captureFrame(video: HTMLVideoElement): ImageData | null {
  const canvas = document.createElement("canvas");
  canvas.width = 160;
  canvas.height = 160;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

function averageBrightness(frame: ImageData): number {
  let total = 0;
  for (let index = 0; index < frame.data.length; index += 16) {
    total += (frame.data[index] * 0.299) + (frame.data[index + 1] * 0.587) + (frame.data[index + 2] * 0.114);
  }
  return total / (frame.data.length / 16);
}

function frameDifference(first: ImageData, second: ImageData): number {
  let difference = 0;
  for (let index = 0; index < first.data.length; index += 16) {
    const firstLuma = (first.data[index] * 0.299) + (first.data[index + 1] * 0.587) + (first.data[index + 2] * 0.114);
    const secondLuma = (second.data[index] * 0.299) + (second.data[index + 1] * 0.587) + (second.data[index + 2] * 0.114);
    difference += Math.abs(firstLuma - secondLuma);
  }
  return difference / (first.data.length / 16);
}

function sharpness(frame: ImageData): number {
  const width = frame.width;
  const lumaAt = (x: number, y: number) => {
    const index = (y * width + x) * 4;
    return (frame.data[index] * 0.299) + (frame.data[index + 1] * 0.587) + (frame.data[index + 2] * 0.114);
  };
  let total = 0;
  let totalSquared = 0;
  let count = 0;
  for (let y = 1; y < frame.height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const edge = Math.abs(lumaAt(x + 1, y) - lumaAt(x - 1, y)) +
        Math.abs(lumaAt(x, y + 1) - lumaAt(x, y - 1));
      total += edge;
      totalSquared += edge * edge;
      count += 1;
    }
  }
  const mean = total / count;
  return (totalSquared / count) - (mean * mean);
}
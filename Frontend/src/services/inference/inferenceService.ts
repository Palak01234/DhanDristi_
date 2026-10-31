import type { DetectionResult, UserPreferences } from "@/types";
import { apiUrl } from "@/services/api";

export interface InferenceOutput {
  result: DetectionResult;
}

export interface FrameQuality {
  ready: boolean;
  message: string;
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

/** Sends a camera frame to the ONNX inference API and maps its response to the UI contract. */
export async function runInference(
  preferences: UserPreferences,
  video?: HTMLVideoElement,
  captureRegion?: HTMLElement,
): Promise<InferenceOutput> {
  if (video && video.videoWidth > 0 && video.videoHeight > 0) {
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
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob) throw new Error("Could not capture a camera frame");

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
      body: blob,
    });
    if (!response.ok) throw new Error(`Model API request failed (${response.status})`);
    return { result: (await response.json()) as DetectionResult };
  }

  throw new Error("Camera frame is not ready. Please wait for the camera preview and try again.");
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
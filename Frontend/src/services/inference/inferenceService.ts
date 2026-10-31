import type { DetectionResult, UserPreferences } from "@/types";
import { apiUrl } from "@/services/api";

export interface InferenceOutput {
  result: DetectionResult;
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
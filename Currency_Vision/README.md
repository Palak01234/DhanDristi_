# Currency Vision

This package contains DhanDrishti's currency-recognition runtime. It supports
both a standalone browser demo and the Node/Express inference API used by the
React application in `../Frontend`.

## Project structure

- `index.html` and `src/` - standalone Vite demo
- `public/model.onnx` - model used for inference
- `public/ort/` - ONNX Runtime WebAssembly files
- `public/samples/` - sample images for testing
- `server.ts` - Express API, authentication, history, TTS proxy, and Node inference
- `dataset.yaml` - 36-class YOLO dataset configuration
- `src/banknotes.ts` - model class map and currency conversion helper

## Run locally

```bash
npm install
npm run build
npm run preview
```

For development:

```bash
npm run dev
```

For the React frontend scanner, run the model API in a second terminal:

```bash
npm run api
```

The API listens on `http://localhost:8787`. The React scanner first validates
the camera frame with ONNX Runtime WebAssembly, then sends the JPEG to
`/api/inference`. The API preprocesses the image with Sharp, runs
`public/model.onnx` with ONNX Runtime Node, and returns detected currencies,
denominations, confidence scores, and timing information.

The frontend uses the local Vite `/api` proxy by default. For a separately
hosted API, set `VITE_API_URL` in `Frontend/.env`.

## Notes

- The standalone demo performs inference locally in the browser.
- The full application requires the API for server inference and MongoDB for
  account and history features.
- Voice synthesis and transcription require a server-side
  `ELEVENLABS_API_KEY`; never expose that key in frontend code.

# Local LLM Mobile (Offline, On-device)

This app runs an LLM fully on Android device using MediaPipe LLM Inference.

- No cloud API calls
- No internet required for inference
- Model is loaded from local device storage
- Built for mobile local execution

## Important note about "without CPU"

On Android, truly zero CPU usage is not possible. The OS and app runtime always use CPU for scheduling and I/O. This project keeps inference local and is designed to use accelerator-friendly model formats (GPU/NNAPI-capable task models), minimizing CPU-heavy cloud/API work.

## What is implemented

- Flutter chat UI in `lib/main.dart`
- Android MethodChannel bridge in `android/app/src/main/kotlin/com/example/flutter_application_1/MainActivity.kt`
- Native local model init/inference/dispose methods:
	- `initModel(modelPath)`
	- `generate(prompt)`
	- `disposeModel()`
- Android dependency `com.google.mediapipe:tasks-genai:0.10.27`

## Device requirements

- Real Android device (emulator is not reliable for LLM inference)
- Android 7.0+ (`minSdk 24`)
- Sufficient RAM/storage for model file

## 1) Get a local `.task` model file

Use a MediaPipe/LiteRT compatible `.task` model.

Example filename used below:

- `model.task`

## 2) Push model to phone using ADB

From your computer terminal:

```bash
adb devices
adb shell rm -rf /data/local/tmp/llm
adb shell mkdir -p /data/local/tmp/llm
adb push model.task /data/local/tmp/llm/model.task
adb shell ls -lh /data/local/tmp/llm
```

## 3) Run the app on device

```bash
flutter pub get
flutter run -d <your_device_id>
```

Find `<your_device_id>` using:

```bash
flutter devices
```

## 4) Use the app

1. Launch app.
2. Keep model path as `/data/local/tmp/llm/model.task` (or change if needed).
3. Tap **Initialize model**.
4. Enter prompt and tap **Send**.

All generation is executed locally on the phone.

## Troubleshooting

- Init fails with file/path error:
	- Verify path with `adb shell ls -lh /data/local/tmp/llm/model.task`
- App opens but generation fails:
	- Ensure model is compatible with MediaPipe LLM Inference.
	- Try a smaller model.
- Slow or crashes on low-end devices:
	- Use a smaller quantized model.
	- Use a newer/high-end phone.

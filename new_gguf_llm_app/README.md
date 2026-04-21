# new_gguf_llm_app

This app runs GGUF LLM inference fully on Android and provides a chat UI for asking questions and receiving responses on device.

## 1) How This App Is Made

This project is built with:

- Flutter (Dart) for UI and app state
- Kotlin Android host (`MainActivity`) connected through `MethodChannel` (`llama_channel`)
- JNI bridge (`llama_jni`) for native llama.cpp calls
- Prebuilt native `.so` libraries in `android/app/src/main/jniLibs/arm64-v8a`
	- `libllama.so`, `libggml.so`, `libggml-cpu.so`, `libggml-base.so`
- Chaquopy (Python on Android) for executing generated Python code and data analysis

High-level flow:

1. Flutter UI scans `.gguf` files from the device model folder.
2. User selects model + prompt style and initializes model.
3. Flutter sends prompt to Kotlin via method channel.
4. Kotlin formats prompt, runs JNI inference, and returns output + token stats.
5. Optional generated Python code is executed with Chaquopy and shown in chat output.

## 2) Main Functions of the App

- Rescan and list local GGUF models.
- Initialize/dispose model in app session.
- Chat inference with token metrics (latency, tokens/sec, token count).
- Prompt style selection:
	- Qwen/ChatML
	- Gemma
	- Generic/Instruct
	- Auto detect
- System prompt customization.
- One-tap sample questions for quick testing.
- Python execution of generated code (with pandas/numpy/matplotlib/seaborn support via Chaquopy).

## 3) How To Add Model Files on Your Mobile for Inference

The app scans this folder on device:

- `/data/local/tmp/gguf_models`

Create folder and push your model:

```bash
adb shell mkdir -p /data/local/tmp/gguf_models
adb push <your_model>.gguf /data/local/tmp/gguf_models/
```

Optional (for dataset-based Python outputs in the app), also push your data files:

```bash
adb push data.pkl /data/local/tmp/gguf_models/preprocessed/
adb push states_data.pkl /data/local/tmp/gguf_models/preprocessed/
adb push ncap_funding_data.pkl /data/local/tmp/gguf_models/preprocessed/
```

Then inside the app:

1. Tap `Rescan files`.
2. Tap `Initialize model`.
3. Choose model and prompt style.
4. Start chatting.

Note: Android build is configured for `arm64-v8a` ABI.

## 4) How To Download/Install This App on Your Mobile

Prerequisites:

- Flutter SDK installed
- Android SDK/NDK installed
- USB debugging enabled on phone

Verify device:

```bash
adb devices
```

### Option A: Run directly from source

```bash
flutter pub get
flutter run -d <device_id>
```

### Option B: Build APK and install

```bash
flutter pub get
flutter build apk --debug
adb install -r build/app/outputs/flutter-apk/app-debug.apk
```

For release install:

```bash
flutter build apk --release
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

package com.example.new_gguf_llm_app

import android.os.Handler
import android.os.Looper
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.IOException
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {

	private val channelName = "llama_channel"
	private val modelDirectory = "/data/local/tmp/gguf_models"
	private val localModelDirectory = "gguf_models"
	private val defaultContextSize = 8192
	private val defaultPredictTokens = 2048

	private val llamaExecutor = Executors.newSingleThreadExecutor()
	private val mainThread = Handler(Looper.getMainLooper())

	private var backendInitialized = false
	private var promptConfig = PromptConfig()
	private var loadedModelPath: String? = null

	private enum class PromptStyle {
		CHATML,
		GEMMA,
		GENERIC,
	}

	companion object {
		init {
			System.loadLibrary("llama_jni")
		}
	}

	private data class PromptConfig(
		val systemPrompt: String = "You are a concise, helpful assistant.",
		val modelName: String = "",
		val promptStyle: PromptStyle = PromptStyle.GENERIC,
	)

	external fun initializeBackend(nativeLibDir: String): String
	external fun initializeModelNative(modelPath: String, nCtx: Int, nPredict: Int): String
	external fun runInferenceNative(prompt: String): String
	external fun getLastGeneratedTokensNative(): Int
	external fun disposeModelNative(): String

	override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
		super.configureFlutterEngine(flutterEngine)

		MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
			.setMethodCallHandler { call, result ->
				when (call.method) {
					"listModels" -> {
						result.success(listModelFiles())
					}

					"initializeModel" -> {
						val modelName = call.argument<String>("modelName")?.trim().orEmpty()
						if (modelName.isEmpty()) {
							result.error("INVALID_MODEL", "Model name is required", null)
							return@setMethodCallHandler
						}

						val systemPrompt = call.argument<String>("systemPrompt")?.trim().orEmpty()
						val modelTypeHint = call.argument<String>("modelType")?.trim().orEmpty()
						val sourceModelPath = resolveModelPath(modelName)

						llamaExecutor.execute {
							try {
								if (!backendInitialized) {
									initializeBackend(applicationInfo.nativeLibraryDir)
									backendInitialized = true
								}

								val modelPath = prepareModelForLoading(sourceModelPath)

								promptConfig = PromptConfig(
									systemPrompt = systemPrompt,
									modelName = modelName,
									promptStyle = inferPromptStyle(modelName, modelTypeHint),
								)

								val output = initializeModelNative(
									modelPath,
									defaultContextSize,
									defaultPredictTokens,
								)

								if (!output.startsWith("Model initialized:")) {
									throw IOException(
										"Native load failed: $output\n" +
											"source: $sourceModelPath\n" +
											"copied: $modelPath",
									)
								}

								loadedModelPath = modelPath

								mainThread.post { result.success(output) }
							} catch (error: Exception) {
								mainThread.post {
									result.error(
										"INIT_ERROR",
										error.message ?: "Failed to initialize model",
										null,
									)
								}
							}
						}
					}

					"runModel" -> {
						val userPrompt = call.argument<String>("prompt")?.trim().orEmpty()
						if (userPrompt.isEmpty()) {
							result.error("INVALID_PROMPT", "Prompt cannot be empty", null)
							return@setMethodCallHandler
						}

						llamaExecutor.execute {
							try {
								val formattedPrompt = composePrompt(promptConfig, userPrompt)
								val output = sanitizeModelOutput(runInferenceNative(formattedPrompt))
								val generatedTokens = getLastGeneratedTokensNative()
								mainThread.post {
									result.success(
										mapOf(
											"text" to output,
											"generatedTokens" to generatedTokens,
										),
									)
								}
							} catch (error: Exception) {
								mainThread.post {
									result.error(
										"INFERENCE_ERROR",
										error.message ?: "Failed to run inference",
										null,
									)
								}
							}
						}
					}

					"runGeneratedCode" -> {
						val code = call.argument<String>("code") ?: ""
						if (code.trim().isEmpty()) {
							result.error("INVALID_CODE", "Generated code cannot be empty", null)
							return@setMethodCallHandler
						}

						llamaExecutor.execute {
							try {
								ensurePythonRuntimeStarted()
								val python = Python.getInstance()
								val module = python.getModule("chaquopy_runner")
								val runResult = module.callAttr("run_code", code, modelDirectory)

								val valuesByKey = mutableMapOf<String, String>()
								for ((keyObj, valueObj) in runResult.asMap()) {
									valuesByKey[keyObj.toString()] = valueObj.toString()
								}

								val okValue = valuesByKey["ok"] ?: "False"
								val isOk =
									okValue.equals("True", ignoreCase = true) ||
										okValue.equals("true", ignoreCase = true)

								mainThread.post {
									result.success(
										mapOf(
											"ok" to isOk,
											"sanitizedCode" to (valuesByKey["sanitizedCode"] ?: code),
											"stdout" to (valuesByKey["stdout"] ?: ""),
											"stderr" to (valuesByKey["stderr"] ?: ""),
											"traceback" to (valuesByKey["traceback"] ?: ""),
										),
									)
								}
							} catch (error: Throwable) {
								mainThread.post {
									result.error(
										"PYTHON_RUN_FAILED",
										error.message ?: "Failed to run Python code",
										error.stackTraceToString(),
									)
								}
							}
						}
					}

					"disposeModel" -> {
						llamaExecutor.execute {
							try {
								val output = disposeModelNative()
								mainThread.post { result.success(output) }
							} catch (error: Exception) {
								mainThread.post {
									result.error(
										"DISPOSE_ERROR",
										error.message ?: "Failed to dispose model",
										null,
									)
								}
							}
						}
					}

					else -> result.notImplemented()
				}
			}
	}

	private fun ensurePythonRuntimeStarted() {
		if (!Python.isStarted()) {
			Python.start(AndroidPlatform(applicationContext))
		}
	}

	private fun resolveModelPath(modelName: String): String {
		if (modelName.startsWith("/")) {
			return modelName
		}
		return "$modelDirectory/$modelName"
	}

	private fun listModelFiles(): List<String> {
		val dir = File(modelDirectory)
		if (!dir.exists() || !dir.isDirectory) {
			return emptyList()
		}

		return dir.listFiles()
			?.asSequence()
			?.filter { it.isFile && it.name.endsWith(".gguf", ignoreCase = true) }
			?.map { it.name }
			?.sorted()
			?.toList()
			?: emptyList()
	}

	private fun prepareModelForLoading(sourcePath: String): String {
		val source = File(sourcePath)
		if (!source.exists()) {
			throw IOException("Model file not found: $sourcePath")
		}
		if (!source.isFile) {
			throw IOException("Model path is not a file: $sourcePath")
		}
		if (!source.canRead()) {
			throw IOException("Model file is not readable by app: $sourcePath")
		}

		val localDir = File(filesDir, localModelDirectory)
		if (!localDir.exists() && !localDir.mkdirs()) {
			throw IOException("Failed to create local model directory: ${localDir.absolutePath}")
		}

		val localModel = File(localDir, source.name)
		val shouldCopy =
			!localModel.exists() ||
				localModel.length() != source.length() ||
				localModel.lastModified() < source.lastModified()

		if (shouldCopy) {
			source.inputStream().use { input ->
				localModel.outputStream().use { output ->
					input.copyTo(output, DEFAULT_BUFFER_SIZE)
				}
			}
		}

		if (!localModel.canRead()) {
			throw IOException("Copied model is not readable: ${localModel.absolutePath}")
		}

		return localModel.absolutePath
	}

	private fun composePrompt(config: PromptConfig, userPrompt: String): String {
		return when (config.promptStyle) {
			PromptStyle.CHATML -> composeChatMlPrompt(config.systemPrompt, userPrompt)
			PromptStyle.GEMMA -> composeGemmaPrompt(config.systemPrompt, userPrompt)
			PromptStyle.GENERIC -> composeGenericPrompt(config.systemPrompt, userPrompt)
		}
	}

	private fun composeChatMlPrompt(
		systemPrompt: String,
		userPrompt: String,
	): String {
		val cleanSystemPrompt = systemPrompt.trim()
		val cleanUserPrompt = userPrompt.trim()

		return buildString {
			append("<|im_start|>system\n")
			append(cleanSystemPrompt)
			append("\n<|im_end|>\n")

			append("<|im_start|>user\n")
			append(cleanUserPrompt)
			append("\n<|im_end|>\n")

			append("<|im_start|>assistant\n")
		}
	}

	private fun composeGemmaPrompt(
		systemPrompt: String,
		userPrompt: String,
	): String {
		val cleanSystemPrompt = systemPrompt.trim()
		val cleanUserPrompt = userPrompt.trim()

		val mergedUserPrompt =
			if (cleanSystemPrompt.isBlank()) {
				cleanUserPrompt
			} else {
				"System instruction:\n$cleanSystemPrompt\n\nUser request:\n$cleanUserPrompt"
			}

		return buildString {
			append("<start_of_turn>user\n")
			append(mergedUserPrompt)
			append("\n<end_of_turn>\n")
			append("<start_of_turn>model\n")
		}
	}

	private fun composeGenericPrompt(
		systemPrompt: String,
		userPrompt: String,
	): String {
		val cleanSystemPrompt = systemPrompt.trim()
		val cleanUserPrompt = userPrompt.trim()

		return buildString {
			if (cleanSystemPrompt.isNotBlank()) {
				append("System:\n")
				append(cleanSystemPrompt)
				append("\n\n")
			}

			append("User:\n")
			append(cleanUserPrompt)
			append("\n\nAssistant:\n")
		}
	}

	private fun inferPromptStyle(modelName: String, modelTypeHint: String): PromptStyle {
		val hint = modelTypeHint.trim().lowercase()
		if (hint.contains("gemma")) {
			return PromptStyle.GEMMA
		}
		if (hint.contains("qwen") || hint.contains("chatml")) {
			return PromptStyle.CHATML
		}
		if (hint.contains("generic") || hint.contains("instruct")) {
			return PromptStyle.GENERIC
		}

		val name = modelName.trim().lowercase()
		if (name.contains("gemma")) {
			return PromptStyle.GEMMA
		}
		if (name.contains("qwen") || name.contains("smol")) {
			return PromptStyle.CHATML
		}

		return PromptStyle.GENERIC
	}

	private fun sanitizeModelOutput(rawOutput: String): String {
		var cleaned = rawOutput

		val thinkBlockRegex = Regex("(?is)<think>.*?</think>")
		cleaned = cleaned.replace(thinkBlockRegex, "")
		cleaned = cleaned.replace("<think>", "", ignoreCase = true)
		cleaned = cleaned.replace("</think>", "", ignoreCase = true)

		cleaned = cleaned.replace("<|im_end|>", "", ignoreCase = true)
		cleaned = cleaned.replace("<end_of_turn>", "", ignoreCase = true)

		cleaned = cleaned
			.replace("```python", "", ignoreCase = true)
			.replace("```", "")
			.trim()

		if (cleaned.isBlank()) {
			return "No output generated."
		}

		val existingFunctionMatch = Regex("(?m)^\\s*def\\s+true_code\\(\\)\\s*:").find(cleaned)
		if (existingFunctionMatch != null) {
			val fromFunction = cleaned.substring(existingFunctionMatch.range.first).trimEnd()
			val lines = fromFunction.lines().toMutableList()
			if (lines.isNotEmpty()) {
				lines[0] = lines[0].trimStart()
			}
			return lines.joinToString("\n").trim()
		}

		val rawLines = cleaned.lines().toMutableList()
		while (rawLines.isNotEmpty() && rawLines.last().isBlank()) {
			rawLines.removeAt(rawLines.lastIndex)
		}

		var hasTerminalCall = false
		if (rawLines.isNotEmpty()) {
			val last = rawLines.last().trim()
			hasTerminalCall = last == "true_code()" || last == "true_code();"
			if (hasTerminalCall) {
				rawLines.removeAt(rawLines.lastIndex)
			}
		}

		val indentedBody = rawLines
			.asSequence()
			.map { line: String ->
				if (line.isBlank()) {
					""
				} else {
					"   ${line.trimEnd()}"
				}
			}
			.joinToString("\n")

		val functionBlock = if (indentedBody.isBlank()) {
			"def true_code():\n    pass"
		} else {
			"def true_code():\n    $indentedBody"
		}

		return if (hasTerminalCall) {
			"$functionBlock\n\ntrue_code()"
		} else {
			functionBlock
		}
	}

	override fun onDestroy() {
		llamaExecutor.execute {
			try {
				disposeModelNative()
				loadedModelPath = null
			} catch (_: Exception) {
			}
		}
		llamaExecutor.shutdown()
		super.onDestroy()
	}
}

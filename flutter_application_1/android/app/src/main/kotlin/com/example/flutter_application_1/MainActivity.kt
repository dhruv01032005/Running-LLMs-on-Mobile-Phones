package com.example.flutter_application_1

import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.embedding.android.FlutterActivity
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import java.io.File
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {
	private val channelName = "local_llm_bridge_channel"
	private val llmExecutor = Executors.newSingleThreadExecutor()
	private val defaultTaskDirectory = "/data/local/tmp/llm"
	private val maxTokens = 1024
	private val defaultTopK = 64
	private val defaultTopP = 0.95f
	private val defaultTemperature = 1.0f
	@Volatile
	private var llmInference: LlmInference? = null
	@Volatile
	private var llmInferenceSession: LlmInferenceSession? = null
	@Volatile
	private var systemPrompt: String = ""
	@Volatile
	private var tokenizerPath: String = ""

	override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
		super.configureFlutterEngine(flutterEngine)

		MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
			.setMethodCallHandler { call: MethodCall, result: MethodChannel.Result ->
				when (call.method) {
					"initModel" -> initModel(call, result)
					"listTaskFiles" -> listTaskFiles(call, result)
					"listTokenizerFiles" -> listTokenizerFiles(call, result)
					"countTokens" -> countTokens(call, result)
					"generate" -> generate(call, result)
					"runGeneratedCode" -> runGeneratedCode(call, result)
					"disposeModel" -> disposeModel(result)
					else -> result.notImplemented()
				}
			}
	}

	private fun listTokenizerFiles(call: MethodCall, result: MethodChannel.Result) {
		val directoryPath = call.argument<String>("directoryPath") ?: defaultTaskDirectory
		llmExecutor.execute {
			try {
				val directory = File(directoryPath)
				if (!directory.exists() || !directory.isDirectory) {
					runOnUiThread {
						result.success(emptyList<String>())
					}
					return@execute
				}

				val tokenizerFiles = directory
					.walkTopDown()
					.filter { file ->
						file.isFile &&
							file.name.endsWith(".json", ignoreCase = true) &&
							file.name.contains("tokenizer", ignoreCase = true)
					}
					.map { file -> file.absolutePath }
					.sorted()
					.toList()

				runOnUiThread {
					result.success(tokenizerFiles)
				}
			} catch (error: Throwable) {
				runOnUiThread {
					result.error("LIST_TOKENIZERS_FAILED", error.message, null)
				}
			}
		}
	}

	private fun listTaskFiles(call: MethodCall, result: MethodChannel.Result) {
		val directoryPath = call.argument<String>("directoryPath") ?: defaultTaskDirectory
		llmExecutor.execute {
			try {
				val directory = File(directoryPath)
				if (!directory.exists() || !directory.isDirectory) {
					runOnUiThread {
						result.success(emptyList<String>())
					}
					return@execute
				}

				val taskFiles = directory
					.walkTopDown()
					.filter { file -> file.isFile && file.name.endsWith(".task", ignoreCase = true) }
					.map { file -> file.absolutePath }
					.sorted()
					.toList()

				runOnUiThread {
					result.success(taskFiles)
				}
			} catch (error: Throwable) {
				runOnUiThread {
					result.error("LIST_TASKS_FAILED", error.message, null)
				}
			}
		}
	}

	private fun initModel(call: MethodCall, result: MethodChannel.Result) {
		val modelPath = call.argument<String>("modelPath")
		val requestedSystemPrompt = call.argument<String>("systemPrompt")?.trim().orEmpty()
		val requestedTokenizerPath = call.argument<String>("tokenizerPath")?.trim().orEmpty()
		if (modelPath.isNullOrBlank()) {
			result.error("INVALID_ARGUMENT", "modelPath is required", null)
			return
		}

		llmExecutor.execute {
			try {
				llmInferenceSession?.close()
				llmInferenceSession = null
				llmInference?.close()
				llmInference = null
				systemPrompt = requestedSystemPrompt
				tokenizerPath = requestedTokenizerPath

				val options = LlmInference.LlmInferenceOptions.builder()
					.setModelPath(modelPath)
					.setMaxTokens(maxTokens)
					.build()

				val inference = LlmInference.createFromOptions(applicationContext, options)
				val session = LlmInferenceSession.createFromOptions(
					inference,
					LlmInferenceSession.LlmInferenceSessionOptions.builder()
						.setTopK(defaultTopK)
						.setTopP(defaultTopP)
						.setTemperature(defaultTemperature)
						.build(),
				)

				llmInference = inference
				llmInferenceSession = session
				runOnUiThread {
					result.success(null)
				}
			} catch (error: Throwable) {
				runOnUiThread {
					result.error("INIT_FAILED", error.message, null)
				}
			}
		}
	}

	private fun countTokens(call: MethodCall, result: MethodChannel.Result) {
		val text = call.argument<String>("text")
		if (text == null) {
			result.error("INVALID_ARGUMENT", "text is required", null)
			return
		}

		llmExecutor.execute {
			try {
				val exactCount = if (tokenizerPath.isNotBlank()) countTokensWithInference(text) else null
				val tokenCount = exactCount ?: fallbackTokenCount(text)
				val fromTokenizer = exactCount != null

				runOnUiThread {
					result.success(
						mapOf(
							"count" to tokenCount,
							"isApproximate" to !fromTokenizer,
						)
					)
				}
			} catch (error: Throwable) {
				runOnUiThread {
					result.error("COUNT_TOKENS_FAILED", error.message, null)
				}
			}
		}
	}

	private fun countTokensWithInference(text: String): Int? {
		val inference = llmInference ?: return null

		val candidateMethodNames = listOf(
			"sizeInTokens",
			"countTokens",
			"computeTokenCount",
			"tokenCount",
		)

		for (methodName in candidateMethodNames) {
			try {
				val method = inference.javaClass.getMethod(methodName, String::class.java)
				val result = method.invoke(inference, text)
				if (result is Number) {
					return result.toInt().coerceAtLeast(0)
				}
			} catch (_: Throwable) {
				// Keep trying alternative method names.
			}
		}

		return null
	}

	private fun fallbackTokenCount(text: String): Int {
		val trimmed = text.trim()
		if (trimmed.isBlank()) {
			return 0
		}
		return trimmed.split(Regex("\\s+")).size
	}

	private fun ensurePythonRuntimeStarted() {
		if (!Python.isStarted()) {
			Python.start(AndroidPlatform(applicationContext))
		}
	}

	private fun runGeneratedCode(call: MethodCall, result: MethodChannel.Result) {
		val code = call.argument<String>("code")
		if (code.isNullOrBlank()) {
			result.error("INVALID_ARGUMENT", "code is required", null)
			return
		}

		llmExecutor.execute {
			try {
				ensurePythonRuntimeStarted()
				val python = Python.getInstance()
				val module = python.getModule("chaquopy_runner")
				val runResult = module.callAttr("run_code", code, defaultTaskDirectory)
				val valuesByKey = mutableMapOf<String, String>()
				for ((keyObj, valueObj) in runResult.asMap()) {
					valuesByKey[keyObj.toString()] = valueObj.toString()
				}
				val okValue = valuesByKey["ok"] ?: "False"
				val isOk = okValue.equals("True", ignoreCase = true) || okValue.equals("true", ignoreCase = true)
				val payload = mapOf(
					"ok" to isOk,
					"sanitizedCode" to (valuesByKey["sanitizedCode"] ?: code),
					"stdout" to (valuesByKey["stdout"] ?: ""),
					"stderr" to (valuesByKey["stderr"] ?: ""),
					"traceback" to (valuesByKey["traceback"] ?: ""),
				)

				runOnUiThread {
					result.success(payload)
				}
			} catch (error: Throwable) {
				runOnUiThread {
					result.error("PYTHON_RUN_FAILED", error.stackTraceToString(), null)
				}
			}
		}
	}

	private fun generateWithSession(session: LlmInferenceSession, prompt: String): String {
		try {
			val generateWithPrompt = session.javaClass.getMethod("generateResponse", String::class.java)
			val result = generateWithPrompt.invoke(session, prompt)
			if (result is String) {
				return result
			}
			if (result != null) {
				return result.toString()
			}
		} catch (_: NoSuchMethodException) {
			// Fall through to older API shape.
		}

		try {
			val addQueryChunk = session.javaClass.getMethod("addQueryChunk", String::class.java)
			addQueryChunk.invoke(session, prompt)
		} catch (_: NoSuchMethodException) {
			throw IllegalStateException(
				"Unsupported LlmInferenceSession API: missing generateResponse(String) and addQueryChunk(String)."
			)
		}

		val generateNoArg = session.javaClass.getMethod("generateResponse")
		val result = generateNoArg.invoke(session)
		return when (result) {
			is String -> result
			null -> ""
			else -> result.toString()
		}
	}

	private fun generate(call: MethodCall, result: MethodChannel.Result) {
		val prompt = call.argument<String>("prompt")
		if (prompt.isNullOrBlank()) {
			result.error("INVALID_ARGUMENT", "prompt is required", null)
			return
		}

		llmExecutor.execute {
			try {
				val session = llmInferenceSession
				if (session == null) {
					runOnUiThread {
						result.error("MODEL_NOT_READY", "Model is not initialized", null)
					}
					return@execute
				}

				val inputPrompt = if (systemPrompt.isBlank()) {
					prompt
				} else {
					"$systemPrompt\n\n$prompt"
				}

				val response = generateWithSession(session, inputPrompt).trimEnd()

				runOnUiThread {
					result.success(response)
				}
			} catch (error: Throwable) {
				runOnUiThread {
					result.error("GENERATION_FAILED", error.message, null)
				}
			}
		}
	}

	private fun disposeModel(result: MethodChannel.Result) {
		llmExecutor.execute {
			try {
				llmInferenceSession?.close()
				llmInferenceSession = null
				llmInference?.close()
				llmInference = null
				systemPrompt = ""
				tokenizerPath = ""
				runOnUiThread {
					result.success(null)
				}
			} catch (error: Throwable) {
				runOnUiThread {
					result.error("DISPOSE_FAILED", error.message, null)
				}
			}
		}
	}

	override fun onDestroy() {
		llmInferenceSession?.close()
		llmInferenceSession = null
		llmInference?.close()
		llmInference = null
		systemPrompt = ""
		tokenizerPath = ""
		llmExecutor.shutdownNow()
		super.onDestroy()
	}
}

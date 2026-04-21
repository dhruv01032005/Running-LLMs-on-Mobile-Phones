import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const GgufMobileApp());
}

class NativeLlamaBridge {
  static const MethodChannel _channel = MethodChannel('llama_channel');

  static Future<List<String>> listModels() async {
    final List<dynamic>? result = await _channel.invokeMethod<List<dynamic>>(
      'listModels',
    );
    return (result ?? <dynamic>[])
        .map((dynamic item) => item.toString())
        .toList();
  }

  static Future<String> initializeModel({
    required String modelName,
    required String systemPrompt,
    required String modelType,
  }) async {
    final String? result = await _channel.invokeMethod<String>(
      'initializeModel',
      <String, dynamic>{
        'modelName': modelName,
        'systemPrompt': systemPrompt,
        'modelType': modelType,
      },
    );
    return result ?? 'Model initialized';
  }

  static Future<GenerationResult> runModel({required String prompt}) async {
    final dynamic result = await _channel.invokeMethod<dynamic>(
      'runModel',
      <String, dynamic>{'prompt': prompt},
    );

    if (result is Map<dynamic, dynamic>) {
      final dynamic textValue = result['text'];
      final dynamic tokenValue = result['generatedTokens'];
      return GenerationResult(
        text: (textValue ?? '').toString(),
        generatedTokens: tokenValue is int ? tokenValue : int.tryParse('$tokenValue'),
      );
    }

    return GenerationResult(text: (result ?? '').toString(), generatedTokens: null);
  }

  static Future<({
    bool ok,
    String sanitizedCode,
    String stdout,
    String stderr,
    String traceback,
  })> runGeneratedCode({required String code}) async {
    final Map<dynamic, dynamic>? result =
        await _channel.invokeMethod<Map<dynamic, dynamic>>(
      'runGeneratedCode',
      <String, dynamic>{'code': code},
    );

    return (
      ok: result?['ok'] as bool? ?? false,
      sanitizedCode: result?['sanitizedCode'] as String? ?? code,
      stdout: result?['stdout'] as String? ?? '',
      stderr: result?['stderr'] as String? ?? '',
      traceback: result?['traceback'] as String? ?? '',
    );
  }

  static Future<String> disposeModel() async {
    final String? result = await _channel.invokeMethod<String>('disposeModel');
    return result ?? 'Model disposed';
  }
}

class GenerationResult {
  const GenerationResult({required this.text, required this.generatedTokens});

  final String text;
  final int? generatedTokens;
}

class ChatMessage {
  const ChatMessage({
    required this.role,
    required this.text,
    this.latencyMs,
    this.tokenCount,
    this.isApproxTokenCount,
    this.tokensPerSecond,
  });

  final String role;
  final String text;
  final int? latencyMs;
  final int? tokenCount;
  final bool? isApproxTokenCount;
  final double? tokensPerSecond;
}

class GgufMobileApp extends StatelessWidget {
  const GgufMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'GGUF Mobile',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0A6A53)),
        useMaterial3: true,
      ),
      home: const GgufChatPage(),
    );
  }
}

class GgufChatPage extends StatefulWidget {
  const GgufChatPage({super.key});

  @override
  State<GgufChatPage> createState() => _GgufChatPageState();
}

class _GgufChatPageState extends State<GgufChatPage> {
  static const String _modelDirectory = '/data/local/tmp/gguf_models';
  static const String _autoModelType = 'Auto detect';
  static const Map<String, String> _modelTypeHints = <String, String>{
    _autoModelType: '',
    'Qwen / ChatML': 'chatml',
    'Gemma': 'gemma',
    'Generic / Instruct': 'generic',
  };
  static const String _defaultSystemPrompt =
  '''You are an AI assistant that writes Python data analysis code to answer questions
using the given datasets.

You must generate a COMPLETE and EXECUTABLE Python program.

Available datasets (already present locally):

1) data.pkl
   - Daily air quality data for India (2017-2024)
   - Columns:
     Timestamp (datetime),
     station (str),
     PM2.5 (float),
     PM10 (float),
     address (str),
     city (str),
     latitude (float),
     longitude (float),
     state (str)

2) states_data.pkl
   - State-wise metadata
   - Columns:
     state (str),
     population (int),
     area (km2) (int),
     isUnionTerritory (bool)

3) ncap_funding_data.pkl
   - NCAP funding information (2019-2022)
   - Columns:
     state (str),
     city (str),
     funding amounts by year,
     total fund released,
     utilisation

Code requirements (IMPORTANT):

- Write all logic inside a function named `true_code()`
- Use pandas, numpy, matplotlib, seaborn as required
- Perform all computations programmatically
- Do NOT print explanations or extra text
- Only Python code must be generated (no markdown, no comments outside code)

Output format:

def true_code():
    ..........
    .....
    print(answer)


true_code()''';
  static const List<String> _sampleQuestions = <String>[
    'Determine the station exhibiting the 3rd highest 25th percentile of PM2.5 in May 2018.',
    'Which state had the 2nd lowest median PM10 during the Monsoon season  of 2024?',
    'Determine which union territory exhibits the highest variance of PM2.5 concentration when considering population density.',
  ];

  final TextEditingController _promptController = TextEditingController();
  final List<String> _modelFiles = <String>[];
  final List<ChatMessage> _messages = <ChatMessage>[];

  String? _selectedModel;
  String _systemPrompt = _defaultSystemPrompt;
  String _selectedModelType = _autoModelType;

  bool _scanning = false;
  bool _initializing = false;
  bool _generating = false;
  bool _modelReady = false;
  String _status = 'Model not initialized';

  String get _statusSummary {
    final String normalized = _status.replaceAll('\n', ' ').trim();
    if (normalized.length <= 150) {
      return normalized;
    }
    return '${normalized.substring(0, 147)}...';
  }

  @override
  void initState() {
    super.initState();
    _refreshModelFiles();
  }

  @override
  void dispose() {
    _promptController.dispose();
    super.dispose();
  }

  Future<void> _refreshModelFiles() async {
    setState(() {
      _scanning = true;
      _status = 'Scanning $_modelDirectory for .gguf files...';
    });

    try {
      final List<String> files = await NativeLlamaBridge.listModels();
      if (!mounted) {
        return;
      }

      setState(() {
        _modelFiles
          ..clear()
          ..addAll(files.where((String f) => f.toLowerCase().endsWith('.gguf')))
          ..sort();

        if (_modelFiles.isEmpty) {
          _selectedModel = null;
          _modelReady = false;
          _status = 'No .gguf files found in $_modelDirectory';
          return;
        }

        if (_selectedModel == null || !_modelFiles.contains(_selectedModel)) {
          _selectedModel = _modelFiles.first;
        }
        _status = 'Found ${_modelFiles.length} model file(s)';
      });
    } on PlatformException catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Scan failed: ${error.message ?? error.code}';
      });
    } finally {
      if (mounted) {
        setState(() {
          _scanning = false;
        });
      }
    }
  }

  Future<void> _openInitializeDialog() async {
    if (_initializing) {
      return;
    }

    if (_modelFiles.isEmpty && !_scanning) {
      await _refreshModelFiles();
    }
    if (!mounted) {
      return;
    }

    String selectedModel = _selectedModel ??
        (_modelFiles.isNotEmpty ? _modelFiles.first : '');
    String draftSystemPrompt = _systemPrompt;
    String draftModelType = _selectedModelType;

    final bool? shouldInitialize = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return StatefulBuilder(
          builder: (BuildContext context, StateSetter setDialogState) {
            return AlertDialog(
              title: const Text('Initialize GGUF Model'),
              content: ConstrainedBox(
                constraints: BoxConstraints(
                  maxHeight: MediaQuery.of(context).size.height * 0.68,
                ),
                child: SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      const Text('Model file'),
                      const SizedBox(height: 8),
                      if (_modelFiles.isNotEmpty)
                        DropdownButtonFormField<String>(
                          isExpanded: true,
                          initialValue: _modelFiles.contains(selectedModel)
                              ? selectedModel
                              : _modelFiles.first,
                          decoration: const InputDecoration(
                            border: OutlineInputBorder(),
                          ),
                          items: _modelFiles
                              .map(
                                (String file) => DropdownMenuItem<String>(
                                  value: file,
                                  child: Text(
                                    _fileNameFromPath(file),
                                    overflow: TextOverflow.ellipsis,
                                    maxLines: 1,
                                  ),
                                ),
                              )
                              .toList(),
                          selectedItemBuilder: (BuildContext context) {
                            return _modelFiles
                                .map(
                                  (String file) => Align(
                                    alignment: Alignment.centerLeft,
                                    child: Text(
                                      _fileNameFromPath(file),
                                      overflow: TextOverflow.ellipsis,
                                      maxLines: 1,
                                    ),
                                  ),
                                )
                                .toList();
                          },
                          onChanged: (String? value) {
                            if (value == null) {
                              return;
                            }
                            setDialogState(() {
                              selectedModel = value;
                            });
                          },
                        )
                      else
                        Text(
                          'No .gguf files found in $_modelDirectory.',
                          style: TextStyle(
                            color: Theme.of(context).colorScheme.error,
                          ),
                        ),
                      const SizedBox(height: 12),
                      const Text('Model type'),
                      const SizedBox(height: 8),
                      DropdownButtonFormField<String>(
                        isExpanded: true,
                        initialValue: _modelTypeHints.containsKey(draftModelType)
                            ? draftModelType
                            : _autoModelType,
                        decoration: const InputDecoration(
                          border: OutlineInputBorder(),
                        ),
                        items: _modelTypeHints.keys
                            .map(
                              (String modelTypeLabel) => DropdownMenuItem<String>(
                                value: modelTypeLabel,
                                child: Text(modelTypeLabel),
                              ),
                            )
                            .toList(),
                        onChanged: (String? value) {
                          if (value == null) {
                            return;
                          }
                          setDialogState(() {
                            draftModelType = value;
                          });
                        },
                      ),
                      const SizedBox(height: 12),
                      const Text('System prompt'),
                      const SizedBox(height: 8),
                      TextFormField(
                        initialValue: draftSystemPrompt,
                        onChanged: (String value) {
                          draftSystemPrompt = value;
                        },
                        minLines: 3,
                        maxLines: 6,
                        decoration: const InputDecoration(
                          border: OutlineInputBorder(),
                          hintText: 'Define assistant behavior',
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              actions: <Widget>[
                TextButton(
                  onPressed: () => Navigator.of(context).pop(false),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: _modelFiles.isEmpty
                      ? null
                      : () => Navigator.of(context).pop(true),
                  child: const Text('Initialize'),
                ),
              ],
            );
          },
        );
      },
    );

    if (shouldInitialize == true) {
      _selectedModel = selectedModel;
      _selectedModelType = draftModelType;
      _systemPrompt = draftSystemPrompt.trim().isEmpty
          ? _defaultSystemPrompt
          : draftSystemPrompt.trim();
      await _initializeSelectedModel();
    }
  }

  Future<void> _initializeSelectedModel() async {
    final String? model = _selectedModel;
    if (model == null || model.isEmpty) {
      setState(() {
        _status = 'Select a .gguf model first';
      });
      return;
    }

    setState(() {
      _initializing = true;
      _status = 'Initializing model...';
    });

    try {
      final String message = await NativeLlamaBridge.initializeModel(
        modelName: model,
        systemPrompt: _systemPrompt,
        modelType: _modelTypeHints[_selectedModelType] ?? '',
      );
      if (!mounted) {
        return;
      }

      setState(() {
        _modelReady = message.startsWith('Model initialized:');
        _status = message;
      });
    } on PlatformException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _modelReady = false;
        _status = 'Initialization failed: ${error.message ?? error.code}';
      });
    } finally {
      if (mounted) {
        setState(() {
          _initializing = false;
        });
      }
    }
  }

  Future<void> _disposeModel() async {
    try {
      final String message = await NativeLlamaBridge.disposeModel();
      if (!mounted) {
        return;
      }

      setState(() {
        _modelReady = false;
        _status = message;
      });
    } on PlatformException catch (error) {
      if (!mounted) {
        return;
      }

      setState(() {
        _status = 'Dispose failed: ${error.message ?? error.code}';
      });
    }
  }

  Future<void> _sendPrompt() async {
    final String prompt = _promptController.text.trim();
    if (prompt.isEmpty || _generating) {
      return;
    }
    if (!_modelReady) {
      setState(() {
        _status = 'Initialize a model before sending prompts';
      });
      return;
    }

    final Stopwatch stopwatch = Stopwatch()..start();

    setState(() {
      _messages.add(ChatMessage(role: 'You', text: prompt));
      _promptController.clear();
      _generating = true;
      _status = 'Generating on-device...';
    });

    try {
      final GenerationResult generation = await NativeLlamaBridge.runModel(prompt: prompt);
      final String output = generation.text;
      late final ({
        bool ok,
        String sanitizedCode,
        String stdout,
        String stderr,
        String traceback,
      }) pythonRunResult;

      try {
        pythonRunResult = await NativeLlamaBridge.runGeneratedCode(code: output);
      } on PlatformException catch (error) {
        pythonRunResult = (
          ok: false,
          sanitizedCode: output,
          stdout: '',
          stderr: '',
          traceback: error.message ?? error.code,
        );
      }

      stopwatch.stop();
      final int tokenCount = generation.generatedTokens ?? _approxTokenCount(output);
      final bool isApproxTokenCount = generation.generatedTokens == null;
      final double elapsedSeconds = stopwatch.elapsedMilliseconds / 1000.0;
      final double tokensPerSecond =
          elapsedSeconds > 0 ? tokenCount / elapsedSeconds : 0;
      final String executionOutput = _formatPythonExecutionOutput(pythonRunResult);

      if (!mounted) {
        return;
      }

      setState(() {
        _messages.add(
          ChatMessage(
            role: 'LLM Code',
            text: pythonRunResult.sanitizedCode.isEmpty
                ? '(empty response)'
                : pythonRunResult.sanitizedCode,
            latencyMs: stopwatch.elapsedMilliseconds,
            tokenCount: tokenCount,
            isApproxTokenCount: isApproxTokenCount,
            tokensPerSecond: tokensPerSecond,
          ),
        );
        _messages.add(ChatMessage(role: 'Expected Answer', text: executionOutput));
        final String executionState =
            pythonRunResult.ok ? 'execution succeeded' : 'execution failed';
        _status =
            'Done in ${stopwatch.elapsedMilliseconds} ms (${isApproxTokenCount ? '≈' : ''}$tokenCount tok, ${tokensPerSecond.toStringAsFixed(2)} tok/s, $executionState)';
      });
    } on PlatformException catch (error) {
      stopwatch.stop();
      if (!mounted) {
        return;
      }

      setState(() {
        _status = 'Inference failed: ${error.message ?? error.code}';
      });
    } finally {
      if (mounted) {
        setState(() {
          _generating = false;
        });
      }
    }
  }

  void _sendSampleQuestion(String question) {
    if (_generating) {
      return;
    }
    _promptController
      ..text = question
      ..selection = TextSelection.collapsed(offset: question.length);
    _sendPrompt();
  }

  Widget _buildSampleQuestionButtons() {
    return SizedBox(
      height: 68,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _sampleQuestions.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (BuildContext context, int index) {
          final String question = _sampleQuestions[index];
          return SizedBox(
            width: 320,
            child: OutlinedButton(
              onPressed: _generating ? null : () => _sendSampleQuestion(question),
              style: OutlinedButton.styleFrom(
                alignment: Alignment.centerLeft,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              ),
              child: Text(
                question,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.left,
              ),
            ),
          );
        },
      ),
    );
  }

  String _formatPythonExecutionOutput(
    ({
      bool ok,
      String sanitizedCode,
      String stdout,
      String stderr,
      String traceback,
    }) runResult,
  ) {
    final String stdout = runResult.stdout.trimRight();
    final String stderr = runResult.stderr.trimRight();
    final String traceback = runResult.traceback.trimRight();

    if (runResult.ok) {
      if (stdout.isNotEmpty) {
        return stdout;
      }
      if (stderr.isNotEmpty) {
        return stderr;
      }
      return '(Code executed with no output)';
    }

    if (traceback.isNotEmpty) {
      return traceback;
    }
    if (stderr.isNotEmpty) {
      return stderr;
    }
    return 'Execution failed with no traceback';
  }

  String _fileNameFromPath(String path) {
    final List<String> parts = path.split(RegExp(r'[\\/]'));
    return parts.isEmpty ? path : parts.last;
  }

  int _approxTokenCount(String text) {
    final String trimmed = text.trim();
    if (trimmed.isEmpty) {
      return 0;
    }
    return trimmed.split(RegExp(r'\s+')).length;
  }

  Widget _buildControlCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'Model directory: $_modelDirectory',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: <Widget>[
                OutlinedButton.icon(
                  onPressed: _scanning ? null : _refreshModelFiles,
                  icon: const Icon(Icons.refresh),
                  label: Text(_scanning ? 'Scanning...' : 'Rescan files'),
                ),
                FilledButton(
                  onPressed: _initializing ? null : _openInitializeDialog,
                  child: Text(_initializing ? 'Initializing...' : 'Initialize model'),
                ),
                OutlinedButton(
                  onPressed: _modelReady ? _disposeModel : null,
                  child: const Text('Dispose model'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text('Found ${_modelFiles.length} model file(s)'),
            const SizedBox(height: 4),
            Text(
              _selectedModel == null
                  ? _status
                  : 'Selected: ${_fileNameFromPath(_selectedModel!)}\nPrompt format: $_selectedModelType\n$_status',
              maxLines: 6,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: _modelReady
                    ? Colors.green.shade700
                    : Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            if (_statusSummary != _status)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  _statusSummary,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 12,
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildMessages() {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.black12),
        borderRadius: BorderRadius.circular(12),
      ),
      child: _messages.isEmpty
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  'Initialize a model and start chatting.',
                  style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant),
                ),
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(12),
              itemCount: _messages.length,
              itemBuilder: (BuildContext context, int index) {
                final ChatMessage message = _messages[index];
                final bool isUser = message.role == 'You';

                return Align(
                  alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: const EdgeInsets.symmetric(vertical: 6),
                    padding: const EdgeInsets.all(10),
                    constraints: BoxConstraints(
                      maxWidth: MediaQuery.of(context).size.width * 0.82,
                    ),
                    decoration: BoxDecoration(
                      color: isUser
                          ? Theme.of(context).colorScheme.primaryContainer
                          : Theme.of(context).colorScheme.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: <Widget>[
                        Text(
                          message.role,
                          style: const TextStyle(fontWeight: FontWeight.w700),
                        ),
                        if (!isUser &&
                            message.latencyMs != null &&
                            message.tokenCount != null &&
                            message.tokensPerSecond != null)
                          Text(
                            '${(message.isApproxTokenCount ?? true) ? '≈' : ''}${message.tokenCount} tok | ${message.tokensPerSecond!.toStringAsFixed(2)} tok/s | ${message.latencyMs} ms',
                            style: TextStyle(
                              fontSize: 12,
                              color: Theme.of(context).colorScheme.onSurfaceVariant,
                            ),
                          ),
                        const SizedBox(height: 4),
                        SelectableText(message.text),
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }

  Widget _buildPromptComposer({required bool showSampleQuestions}) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: <Widget>[
        if (showSampleQuestions) ...<Widget>[
          Align(
            alignment: Alignment.centerLeft,
            child: Text(
              'Sample questions',
              style: Theme.of(context).textTheme.labelLarge,
            ),
          ),
          const SizedBox(height: 6),
          _buildSampleQuestionButtons(),
          const SizedBox(height: 8),
        ],
        Row(
          children: <Widget>[
            Expanded(
              child: TextField(
                controller: _promptController,
                enabled: !_generating,
                textInputAction: TextInputAction.send,
                keyboardType: TextInputType.multiline,
                onSubmitted: (_) => _sendPrompt(),
                minLines: 1,
                maxLines: 3,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  hintText: 'Type your prompt',
                ),
              ),
            ),
            const SizedBox(width: 8),
            FilledButton(
              onPressed: _generating ? null : _sendPrompt,
              child: Text(_generating ? '...' : 'Send'),
            ),
          ],
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final MediaQueryData media = MediaQuery.of(context);
    final bool keyboardVisible = media.viewInsets.bottom > 0;

    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: AppBar(title: const Text('GGUF Mobile (llama.cpp)')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
          child: Column(
            children: <Widget>[
              if (!keyboardVisible) ...<Widget>[
                _buildControlCard(),
                const SizedBox(height: 8),
              ],
              Expanded(child: _buildMessages()),
            ],
          ),
        ),
      ),
      bottomNavigationBar: AnimatedPadding(
        duration: const Duration(milliseconds: 180),
        curve: Curves.easeOut,
        padding: EdgeInsets.fromLTRB(12, 8, 12, 12 + media.viewInsets.bottom),
        child: SafeArea(
          top: false,
          child: _buildPromptComposer(showSampleQuestions: !keyboardVisible),
        ),
      ),
    );
  }
}

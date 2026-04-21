import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const LocalLlmApp());
}

class LocalLlmApp extends StatelessWidget {
  const LocalLlmApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Local LLM Mobile',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0B6E4F)),
      ),
      home: const LocalLlmPage(),
    );
  }
}

class LocalLlmBridge {
  static const MethodChannel _channel = MethodChannel(
    'local_llm_bridge_channel',
  );

  static Future<void> initModel({
    required String modelPath,
    required String systemPrompt,
    required String tokenizerPath,
  }) async {
    await _channel.invokeMethod<void>('initModel', <String, dynamic>{
      'modelPath': modelPath,
      'systemPrompt': systemPrompt,
      'tokenizerPath': tokenizerPath,
    });
  }

  static Future<List<String>> listTaskFiles({
    String directoryPath = '/data/local/tmp/llm',
  }) async {
    final List<dynamic>? result = await _channel.invokeMethod<List<dynamic>>(
      'listTaskFiles',
      <String, dynamic>{'directoryPath': directoryPath},
    );
    return (result ?? <dynamic>[]).cast<String>();
  }

  static Future<List<String>> listTokenizerFiles({
    String directoryPath = '/data/local/tmp/llm',
  }) async {
    final List<dynamic>? result = await _channel.invokeMethod<List<dynamic>>(
      'listTokenizerFiles',
      <String, dynamic>{'directoryPath': directoryPath},
    );
    return (result ?? <dynamic>[]).cast<String>();
  }

  static Future<({int count, bool isApproximate})> countTokens({
    required String text,
  }) async {
    final Map<dynamic, dynamic>? result = await _channel
        .invokeMethod<Map<dynamic, dynamic>>('countTokens', <String, dynamic>{
          'text': text,
        });

    final int count = (result?['count'] as num?)?.toInt() ?? 0;
    final bool isApproximate = result?['isApproximate'] as bool? ?? true;
    return (count: count, isApproximate: isApproximate);
  }

  static Future<String> generate({required String prompt}) async {
    final String? result = await _channel.invokeMethod<String>(
      'generate',
      <String, dynamic>{'prompt': prompt},
    );
    return result ?? '';
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

  static Future<void> disposeModel() async {
    await _channel.invokeMethod<void>('disposeModel');
  }
}

class ChatMessage {
  const ChatMessage({
    required this.role,
    required this.text,
    this.approxTokens,
    this.tokensPerSecond,
    this.latencyMs,
    this.isTokenEstimateApproximate,
  });

  final String role;
  final String text;
  final int? approxTokens;
  final double? tokensPerSecond;
  final int? latencyMs;
  final bool? isTokenEstimateApproximate;
}

class LocalLlmPage extends StatefulWidget {
  const LocalLlmPage({super.key});

  @override
  State<LocalLlmPage> createState() => _LocalLlmPageState();
}

class _LocalLlmPageState extends State<LocalLlmPage> {
  static const String _taskDirectory = '/data/local/tmp/llm';
  static const String _defaultSystemPrompt = '''You are an AI assistant that writes Python data analysis code to answer questions
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

  final TextEditingController _modelPathController = TextEditingController(
    text: '/data/local/tmp/llm/model.task',
  );
  final TextEditingController _promptController = TextEditingController();
  final List<ChatMessage> _messages = <ChatMessage>[];
  final List<String> _taskFiles = <String>[];
  final List<String> _tokenizerFiles = <String>[];

  String? _selectedTaskFile;
  String? _selectedTokenizerFile;
  String _systemPrompt = _defaultSystemPrompt;
  bool _initializing = false;
  bool _loadingTaskFiles = false;
  bool _generating = false;
  bool _modelReady = false;
  String _status = 'Model not initialized';

  @override
  void initState() {
    super.initState();
    _refreshTaskFiles();
  }

  @override
  void dispose() {
    _modelPathController.dispose();
    _promptController.dispose();
    super.dispose();
  }

  Future<void> _refreshTaskFiles() async {
    setState(() {
      _loadingTaskFiles = true;
      _status = 'Scanning $_taskDirectory for .task files...';
    });

    try {
      final List<String> files = await LocalLlmBridge.listTaskFiles(
        directoryPath: _taskDirectory,
      );
      final List<String> tokenizers = await LocalLlmBridge.listTokenizerFiles(
        directoryPath: _taskDirectory,
      );
      if (!mounted) {
        return;
      }
      setState(() {
        _taskFiles
          ..clear()
          ..addAll(files);
        if (_taskFiles.isNotEmpty) {
          _selectedTaskFile = _taskFiles.first;
          _modelPathController.text = _selectedTaskFile!;
          _status = 'Found ${_taskFiles.length} model file(s)';
        } else {
          _selectedTaskFile = null;
          _status = 'No .task files found in $_taskDirectory';
        }

        _tokenizerFiles
          ..clear()
          ..addAll(tokenizers);
        _selectedTokenizerFile = _tokenizerFiles.isNotEmpty
            ? _tokenizerFiles.first
            : null;
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
          _loadingTaskFiles = false;
        });
      }
    }
  }

  Future<void> _openInitDialog() async {
    if (_initializing) {
      return;
    }

    if (_taskFiles.isEmpty && !_loadingTaskFiles) {
      await _refreshTaskFiles();
    }

    if (!mounted) {
      return;
    }

    String selectedPath = _selectedTaskFile ?? _modelPathController.text.trim();
    String? selectedTokenizerPath = _selectedTokenizerFile;
    final TextEditingController systemPromptController = TextEditingController(
      text: _systemPrompt.isEmpty ? _defaultSystemPrompt : _systemPrompt,
    );

    final bool? shouldInit = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return StatefulBuilder(
          builder: (BuildContext context, StateSetter setDialogState) {
            return AlertDialog(
              title: const Text('Initialize model'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    const Text('Select model file'),
                    const SizedBox(height: 8),
                    if (_taskFiles.isNotEmpty)
                      DropdownButtonFormField<String>(
                        isExpanded: true,
                        initialValue: _taskFiles.contains(selectedPath)
                            ? selectedPath
                            : _taskFiles.first,
                        decoration: const InputDecoration(
                          border: OutlineInputBorder(),
                        ),
                        items: _taskFiles
                            .map(
                              (String path) => DropdownMenuItem<String>(
                                value: path,
                                child: Text(
                                  _fileNameFromPath(path),
                                  overflow: TextOverflow.ellipsis,
                                  maxLines: 1,
                                ),
                              ),
                            )
                            .toList(),
                        selectedItemBuilder: (BuildContext context) {
                          return _taskFiles
                              .map(
                                (String path) => Align(
                                  alignment: Alignment.centerLeft,
                                  child: Text(
                                    _fileNameFromPath(path),
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
                            selectedPath = value;
                          });
                        },
                      ),
                    if (_taskFiles.isEmpty)
                      Text(
                        'No .task files found in $_taskDirectory. Tap Rescan files, then reopen Initialize model.',
                        style: TextStyle(
                          color: Theme.of(context).colorScheme.onSurfaceVariant,
                        ),
                      ),
                    const SizedBox(height: 12),
                    const Text('Tokenizer (optional)'),
                    const SizedBox(height: 8),
                    DropdownButtonFormField<String?>(
                      isExpanded: true,
                      initialValue: selectedTokenizerPath,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                      ),
                      items: <DropdownMenuItem<String?>>[
                        const DropdownMenuItem<String?>(
                          value: null,
                          child: Text('None'),
                        ),
                        ..._tokenizerFiles.map(
                          (String path) => DropdownMenuItem<String?>(
                            value: path,
                            child: Text(
                              _fileNameFromPath(path),
                              overflow: TextOverflow.ellipsis,
                              maxLines: 1,
                            ),
                          ),
                        ),
                      ],
                      selectedItemBuilder: (BuildContext context) {
                        return <Widget>[
                          const Align(
                            alignment: Alignment.centerLeft,
                            child: Text('None'),
                          ),
                          ..._tokenizerFiles.map(
                            (String path) => Align(
                              alignment: Alignment.centerLeft,
                              child: Text(
                                _fileNameFromPath(path),
                                overflow: TextOverflow.ellipsis,
                                maxLines: 1,
                              ),
                            ),
                          ),
                        ];
                      },
                      onChanged: (String? value) {
                        setDialogState(() {
                          selectedTokenizerPath = value;
                        });
                      },
                    ),
                    const SizedBox(height: 12),
                    const Text('System prompt (optional)'),
                    const SizedBox(height: 8),
                    TextField(
                      controller: systemPromptController,
                      maxLines: 4,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        hintText: 'Define assistant behavior and constraints',
                      ),
                    ),
                  ],
                ),
              ),
              actions: <Widget>[
                TextButton(
                  onPressed: () => Navigator.of(context).pop(false),
                  child: const Text('Cancel'),
                ),
                FilledButton(
                  onPressed: _taskFiles.isEmpty
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

    if (shouldInit == true) {
      final String promptInput = systemPromptController.text.trim();
      _systemPrompt = promptInput.isEmpty ? _defaultSystemPrompt : promptInput;
      _modelPathController.text = selectedPath;
      _selectedTaskFile = selectedPath.isEmpty ? null : selectedPath;
      _selectedTokenizerFile = selectedTokenizerPath;
      await _initModel();
    }

    systemPromptController.dispose();
  }

  Future<void> _initModel() async {
    setState(() {
      _initializing = true;
      _status = 'Initializing model...';
    });

    try {
      final String effectiveSystemPrompt = _systemPrompt.trim().isEmpty
          ? _defaultSystemPrompt
          : _systemPrompt.trim();
      await LocalLlmBridge.initModel(
        modelPath: _modelPathController.text.trim(),
        systemPrompt: effectiveSystemPrompt,
        tokenizerPath: _selectedTokenizerFile ?? '',
      );
      _systemPrompt = effectiveSystemPrompt;
      if (!mounted) {
        return;
      }
      setState(() {
        _modelReady = true;
        if (_systemPrompt.isNotEmpty) {
          _status = 'Model ready with system prompt (target max: 1024 tokens)';
        } else {
          _status = 'Model ready (target max: 1024 tokens)';
        }
      });
    } on PlatformException catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _modelReady = false;
        _status = 'Init failed: ${error.message ?? error.code}';
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
      await LocalLlmBridge.disposeModel();
      if (!mounted) {
        return;
      }
      setState(() {
        _modelReady = false;
        _status = 'Model disposed';
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
        _status = 'Initialize model before sending prompts';
      });
      return;
    }

    final Stopwatch stopwatch = Stopwatch()..start();

    setState(() {
      _messages.add(ChatMessage(role: 'You', text: prompt));
      _promptController.clear();
      _generating = true;
      _status = 'Generating response on-device...';
    });

    try {
      final String answer = await LocalLlmBridge.generate(prompt: prompt);
      stopwatch.stop();

      final ({int count, bool isApproximate}) tokenStats =
          await LocalLlmBridge.countTokens(text: answer);
      final ({
        bool ok,
        String sanitizedCode,
        String stdout,
        String stderr,
        String traceback,
      }) pythonRunResult = await LocalLlmBridge.runGeneratedCode(code: answer);

      final int tokenCount = tokenStats.count;
      final double elapsedSeconds = stopwatch.elapsedMilliseconds / 1000.0;
      final double tokensPerSecond = elapsedSeconds > 0
          ? tokenCount / elapsedSeconds
          : 0;

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
            approxTokens: tokenCount,
            tokensPerSecond: tokensPerSecond,
            latencyMs: stopwatch.elapsedMilliseconds,
            isTokenEstimateApproximate: tokenStats.isApproximate,
          ),
        );
        _messages.add(ChatMessage(role: 'Expected Answer', text: executionOutput));
        final String metricLabel = tokenStats.isApproximate
            ? 'approx tokens'
            : 'tokens';
        final String executionState = pythonRunResult.ok
            ? 'execution succeeded'
            : 'execution failed';
        _status =
            'Done (${tokenCount.toString()} $metricLabel, ${tokensPerSecond.toStringAsFixed(2)} tok/s, $executionState)';
      });
    } on PlatformException catch (error) {
      stopwatch.stop();
      if (!mounted) {
        return;
      }
      setState(() {
        _status = 'Generate failed: ${error.message ?? error.code}';
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
    final List<String> parts = path.split('/');
    return parts.isEmpty ? path : parts.last;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Local LLM (Offline, On-device)')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            children: <Widget>[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: <Widget>[
                          OutlinedButton.icon(
                            onPressed: _loadingTaskFiles
                                ? null
                                : _refreshTaskFiles,
                            icon: const Icon(Icons.refresh),
                            label: Text(
                              _loadingTaskFiles
                                  ? 'Scanning...'
                                  : 'Rescan files',
                            ),
                          ),
                          FilledButton(
                            onPressed: _initializing ? null : _openInitDialog,
                            child: Text(
                              _initializing
                                  ? 'Initializing...'
                                  : 'Initialize model',
                            ),
                          ),
                          OutlinedButton(
                            onPressed: _disposeModel,
                            child: const Text('Dispose model'),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _status,
                        style: TextStyle(
                          color: _modelReady
                              ? Colors.green.shade700
                              : Colors.orange.shade800,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 8),
              Expanded(
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.black12),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: ListView.builder(
                    padding: const EdgeInsets.all(12),
                    itemCount: _messages.length,
                    itemBuilder: (BuildContext context, int index) {
                      final ChatMessage message = _messages[index];
                      final bool isUser = message.role == 'You';
                      return Align(
                        alignment: isUser
                            ? Alignment.centerRight
                            : Alignment.centerLeft,
                        child: Container(
                          margin: const EdgeInsets.symmetric(vertical: 6),
                          padding: const EdgeInsets.all(10),
                          constraints: const BoxConstraints(maxWidth: 320),
                          decoration: BoxDecoration(
                            color: isUser
                                ? Theme.of(context).colorScheme.primaryContainer
                                : Theme.of(
                                    context,
                                  ).colorScheme.surfaceContainerHighest,
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: <Widget>[
                              Text(
                                message.role,
                                style: const TextStyle(
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                                if (message.role == 'LLM Code' &&
                                  message.tokensPerSecond != null &&
                                  message.approxTokens != null &&
                                  message.latencyMs != null)
                                Text(
                                  '${(message.isTokenEstimateApproximate ?? true) ? '≈' : ''}${message.approxTokens} tok | ${message.tokensPerSecond!.toStringAsFixed(2)} tok/s | ${message.latencyMs} ms',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Theme.of(
                                      context,
                                    ).colorScheme.onSurfaceVariant,
                                  ),
                                ),
                              const SizedBox(height: 4),
                              Text(message.text),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ),
              const SizedBox(height: 8),
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
              Row(
                children: <Widget>[
                  Expanded(
                    child: TextField(
                      controller: _promptController,
                      enabled: !_generating,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _sendPrompt(),
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        hintText: 'Type a question',
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
          ),
        ),
      ),
    );
  }
}

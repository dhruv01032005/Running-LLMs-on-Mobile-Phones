import 'package:flutter_test/flutter_test.dart';

import 'package:flutter_application_1/main.dart';

void main() {
  testWidgets('Local app renders model controls', (WidgetTester tester) async {
    await tester.pumpWidget(const LocalLlmApp());

    expect(find.text('Local LLM (Offline, On-device)'), findsOneWidget);
    expect(find.text('Initialize model'), findsOneWidget);
    expect(find.text('Dispose model'), findsOneWidget);
  });
}

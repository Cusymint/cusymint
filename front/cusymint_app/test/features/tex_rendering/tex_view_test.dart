import 'package:cusymint_app/features/tex_rendering/pages/example_integrals_page.dart';
import 'package:golden_toolkit/golden_toolkit.dart';

void main() {
  testGoldens('TexView group test', (tester) async {
    final builder = DeviceBuilder()
      ..addScenario(widget: const ExampleIntegralsPage());

    await tester.pumpDeviceBuilder(builder);

    await screenMatchesGolden(tester, 'tex_view_group_test');
  });
}

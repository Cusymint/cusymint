import 'package:cusymint_ui/cusymint_ui.dart';

class ExampleIntegralsPage extends StatelessWidget {
  const ExampleIntegralsPage({super.key});

  @override
  Widget build(BuildContext context) {
    const scale = 2.0;

    return Scaffold(
      body: Center(
        child: Column(
          children: const [
            TexView(
              r'''\int_{a}^{b} x^2 \, dx''',
              fontScale: scale,
            ),
            TexView(
              r'''\int_{a}^{b} e^x + x^{x^{x^{x^x}}} - \sqrt[3]{15x} \, dx''',
              fontScale: scale,
            ),
            Divider(),
            TexView(
              r'''\int(x^{2} + x - 1) \, dx''',
              fontScale: scale,
            ),
            TexView(
              r'''\int e^{2x} \, dx''',
              fontScale: scale,
            ),
            TexView(
              r'''''',
              fontScale: scale,
            ),
            TexView(
              r'''\int \frac{1}{x} \, dx''',
              fontScale: scale,
            ),
            TexView(
              r'''\int \frac{1}{x^2} \, dx''',
              fontScale: scale,
            ),
            TexView(
              r'''\int \sin^2(x) \, dx''',
              fontScale: scale,
            ),
            Divider(),
            TexView(
              r'''\int \ln \frac{\tg{x^{\arctg{x}}}}{\sqrt{\cos x}} \, dx''',
              fontScale: scale,
            ),
          ],
        ),
      ),
    );
  }
}

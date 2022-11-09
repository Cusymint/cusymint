import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class AboutTemplate extends StatelessWidget {
  const AboutTemplate({super.key});

  final List<String> authors = const [
    'Szymon Tur',
    'Dawid Wysocki',
    'Szymon Zyguła',
    'Krzysztof Kaczmarski'
  ];

  @override
  Widget build(BuildContext context) {
    return CuScaffold(
      appBar: CuAppBar(
        hasLogo: false,
      ),
      body: Center(
        child: Column(
          children: [
            const CuLogo(),
            const SizedBox(height: 20),
            CuText.med14(Strings.madeBy.tr()),
            const SizedBox(height: 6),
            for (final author in authors) CuText.med14(author),
          ],
        ),
      ),
    );
  }
}

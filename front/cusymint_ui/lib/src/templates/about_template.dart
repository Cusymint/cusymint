import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/gestures.dart';

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
    final colors = CuColors.of(context);

    return CuScaffold(
      appBar: CuAppBar(),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              CuText.bold14(
                Strings.subtitle.tr(),
                color: colors.textWhite,
              ),
              const SizedBox(height: 16),
              RichText(
                text: TextSpan(
                  children: <TextSpan>[
                    TextSpan(
                      text: '${Strings.aboutText.tr()} ',
                      style: CuTextStyle.med14(color: colors.textBlack),
                    ),
                    TextSpan(
                      text: Strings.aboutTextGithub.tr(),
                      style: CuTextStyle.med14(color: colors.mintDark),
                      recognizer: TapGestureRecognizer()..onTap = () {},
                    ),
                    TextSpan(
                      text: '.',
                      style: CuTextStyle.med14(color: colors.textBlack),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              CuText.med14(Strings.madeBy.tr()),
              const SizedBox(height: 6),
              for (final author in authors) CuText.med14(author),
            ],
          ),
        ),
      ),
    );
  }
}

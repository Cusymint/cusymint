import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuWelcomePageTemplate extends StatelessWidget {
  const CuWelcomePageTemplate({
    super.key,
    this.onTextFieldTap,
    this.drawer,
    this.texCards = const [],
  });

  final VoidCallback? onTextFieldTap;
  final CuDrawer? drawer;
  final List<CuTexCard> texCards;

  @override
  Widget build(BuildContext context) {
    return CuScaffold(
      appBar: CuAppBar(hasLogo: false),
      drawer: drawer,
      body: Center(
        child: Column(
          children: [
            Expanded(
              flex: 2,
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Hero(tag: 'logo', child: CuLogo()),
                    CuText.bold14(
                      Strings.subtitle.tr(),
                      color: CuColors.of(context).white,
                    ),
                  ],
                ),
              ),
            ),
            Expanded(
              flex: 3,
              child: Column(
                children: [
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: SizedBox(
                      width: 400,
                      child: Hero(
                        tag: 'input',
                        child: Column(
                          children: [
                            CuText.med14(
                              Strings.enterIntegral.tr(),
                            ),
                            CuTextField(
                              onTap: onTextFieldTap,
                              keyboardType: TextInputType.none,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  CuExampleIntegralsRow(
                    texCards: texCards,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

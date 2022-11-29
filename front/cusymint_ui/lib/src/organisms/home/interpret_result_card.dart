import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:cusymint_ui/src/organisms/home/base_card.dart';

class CuInterpretResultCard extends StatelessWidget {
  const CuInterpretResultCard({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return BaseCard(
      title: CuText.med14(Strings.interpretedAs.tr()),
      children: [
        CuScrollableHorizontalWrapper(
          child: child,
        ),
      ],
    );
  }
}

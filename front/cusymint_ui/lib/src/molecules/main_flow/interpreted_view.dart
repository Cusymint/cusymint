import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuInterpretedView extends StatelessWidget {
  const CuInterpretedView({super.key, this.child});

  final Widget? child;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        CuText.med14(Strings.interpretedAs.tr()),
        CuCard(child: child),
      ],
    );
  }
}

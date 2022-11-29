import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:cusymint_ui/src/organisms/home/base_card.dart';

class CuInterpretLoadingCard extends StatelessWidget {
  const CuInterpretLoadingCard({super.key});

  @override
  Widget build(BuildContext context) {
    return BaseCard(
      title: CuTextLoading(Strings.interpreting.tr()),
      children: const [
        CuScrollableHorizontalWrapper(
          child: CuShimmerRectangle(
            height: 50,
            width: 400,
          ),
        ),
      ],
    );
  }
}

import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuAboutWidget extends StatelessWidget {
  const CuAboutWidget({
    Key? key,
    required this.onGithubTap,
  }) : super(key: key);

  final VoidCallback onGithubTap;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return RichText(
      text: CuTextSpan.med14(
        color: colors.black,
        children: <TextSpan>[
          CuTextSpan(text: '${Strings.aboutText.tr()} '),
          CuTextSpan.link14(
            text: Strings.aboutTextGithub.tr(),
            onTap: onGithubTap,
            color: colors.mintHeavyish,
          ),
          const CuTextSpan(text: '.'),
        ],
      ),
    );
  }
}

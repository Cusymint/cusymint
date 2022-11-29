import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:cusymint_ui/src/organisms/home/base_card.dart';

class CuSolveResultCard extends StatelessWidget {
  const CuSolveResultCard({
    super.key,
    required this.solvingDuration,
    required this.shareUtf,
    required this.copyUtf,
    required this.copyTex,
    required this.child,
  });

  final Widget child;

  final Duration solvingDuration;

  final VoidCallback shareUtf;
  final VoidCallback copyUtf;
  final VoidCallback copyTex;

  @override
  Widget build(BuildContext context) {
    return BaseCard(
      title: CuText.med14(
        Strings.foundResult.tr(namedArgs: {
          'timeInMs': solvingDuration.inMilliseconds.toString(),
        }),
      ),
      children: [
        CuScrollableHorizontalWrapper(
          child: child,
        ),
        _ResultButtons(
          copyTex: copyTex,
          copyUtf: copyUtf,
          shareUtf: shareUtf,
        ),
      ],
    );
  }
}

class _ResultButtons extends StatelessWidget {
  const _ResultButtons({
    super.key,
    required this.shareUtf,
    required this.copyUtf,
    required this.copyTex,
  });

  final VoidCallback shareUtf;
  final VoidCallback copyUtf;
  final VoidCallback copyTex;

  @override
  Widget build(BuildContext context) {
    // TODO: replace with cusymint icons
    return ButtonBar(
      alignment: MainAxisAlignment.end,
      children: [
        IconButton(
          onPressed: shareUtf,
          icon: const Icon(Icons.share),
        ),
        IconButton(
          onPressed: copyTex,
          icon: const Icon(Icons.copy),
        ),
        IconButton(
          onPressed: copyUtf,
          icon: const Icon(Icons.copy_sharp),
        ),
      ],
    );
  }
}

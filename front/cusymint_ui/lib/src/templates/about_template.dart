import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class AboutTemplate extends StatelessWidget {
  const AboutTemplate({
    super.key,
    required this.onGithubTap,
  });

  final VoidCallback onGithubTap;

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
              CuAboutWidget(onGithubTap: onGithubTap),
              const SizedBox(height: 20),
              const CuAuthorsWidget(),
            ],
          ),
        ),
      ),
    );
  }
}

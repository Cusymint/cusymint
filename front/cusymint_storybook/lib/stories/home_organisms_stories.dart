import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class HomeOrganismsStories extends StorybookPart {
  const HomeOrganismsStories();

  @override
  List<Story> get stories => [
        Story(
          name: 'Organisms/Home/InterpretLoadingCard',
          builder: (context) => const CuInterpretLoadingCard(),
        ),
        Story(
          name: 'Organisms/Home/SolveResultCard',
          builder: (context) => CuSolveResultCard(
            solvingDuration: const Duration(milliseconds: 328),
            copyTex: () {},
            copyUtf: () {},
            shareUtf: () {},
            child: Placeholder(
              fallbackHeight: 60,
              fallbackWidth: context.knobs.slider(
                label: 'Result width',
                initial: 100,
                max: 10000,
                min: 10,
              ),
            ),
          ),
        ),
      ];
}

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
          name: 'Organisms/Home/InterpretResultCard',
          builder: (context) => const CuInterpretResultCard(
            child: Placeholder(
              fallbackHeight: 50,
              fallbackWidth: 400,
            ),
          ),
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
        Story(
          name: 'Organisms/Home/AnimatedHomeCard',
          builder: (context) {
            final hasAllCallbacks = context.knobs.boolean(
              label: 'Has all callbacks',
              initial: true,
            );

            final hasErrors = context.knobs.boolean(
              label: 'Has errors',
              initial: false,
            );

            return CuAnimatedHomeCard(
              title: context.knobs.nullable.text(
                label: 'Leading',
                initial: 'Leading',
              ),
              errors: hasErrors ? ['Error 1', 'Error 2', 'Error 3'] : [],
              hasCriticalErrors: context.knobs.boolean(
                label: 'Has critical errors',
                initial: false,
              ),
              inputInTex: context.knobs.text(
                label: 'Input in TeX',
                initial: '\\int 15\\text{d}x',
              ),
              outputInTex: context.knobs.text(
                label: 'Output in TeX',
                initial: '\\int 7.5x\\text{d}x',
              ),
              onCopyUtf: hasAllCallbacks ? () {} : null,
              onCopyTex: hasAllCallbacks ? () {} : null,
              onShareUtf: hasAllCallbacks ? () {} : null,
            );
          },
        ),
      ];
}

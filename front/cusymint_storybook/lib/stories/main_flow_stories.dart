import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class MainFlowStories extends StorybookPart {
  const MainFlowStories();

  @override
  List<Story> get stories => [
        Story(
          name: 'MainFlow/CalculatingView',
          builder: (context) => const CuCalculatingView(),
        ),
        Story(
          name: 'MainFlow/InterpretingView',
          builder: (context) => const CuInterpretingView(),
        ),
        Story(
          name: 'MainFlow/ResultView',
          builder: (context) {
            final repeatTextCount = context.knobs.sliderInt(
              label: 'Repeat Text',
              min: 1,
              max: 10,
              initial: 1,
            );

            return CuResultView(
              duration: const Duration(milliseconds: 857),
              copyTexToClipboard: () {},
              shareUtf: () {},
              copyUtfToClipboard: () {},
              child: CuText.med24('∫123 + x dx * 13' * repeatTextCount),
            );
          },
        ),
        Story(
          name: 'MainFlow/InterpretedView',
          builder: (context) {
            final repeatTextCount = context.knobs.sliderInt(
              label: 'Repeat Text',
              min: 1,
              max: 10,
              initial: 1,
            );

            return CuInterpretedView(
              child: CuText.med24('∫123 + x dx * 13' * repeatTextCount),
            );
          },
        ),
      ];
}

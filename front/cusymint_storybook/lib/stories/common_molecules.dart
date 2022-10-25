import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class CommonMolecules extends StorybookPart {
  const CommonMolecules();

  @override
  List<Story> get stories => [
        Story(
          name: 'Molecules/Carousel',
          builder: (context) => SizedBox(
            height: 100,
            child: CuCarousel(
              children: [
                for (int i = 0;
                    i <
                        context.knobs.sliderInt(
                          label: 'Children count',
                          initial: 5,
                        );
                    ++i)
                  CuCard(
                    child: CuText('Child #${i + 1}'),
                  ),
              ],
            ),
          ),
        ),
        Story(
          name: 'Molecules/Drawer',
          builder: (context) => CuScaffold(
            body: const Center(child: CuText('Drawer')),
            drawer: CuDrawer(
              onAboutPressed: () {},
              onHomePressed: () {},
              onSettingsPressed: () {},
            ),
            appBar: CuAppBar(),
          ),
        ),
        Story(
          name: 'Molecules/TextLoading',
          builder: (context) => CuTextLoading.med14(
            context.knobs.text(
              label: 'Text',
              initial: 'Loading',
            ),
          ),
        ),
        Story(
          name: 'Molecules/Scaffold',
          builder: (context) {
            final displayAppBar = context.knobs.boolean(
              label: 'Display app bar',
              initial: true,
            );

            final displayMenuButton = context.knobs.boolean(
              label: 'Display menu button',
              initial: true,
            );

            final appBar = displayAppBar
                ? CuAppBar(
                    onMenuPressed: displayMenuButton ? () {} : null,
                  )
                : null;

            return CuScaffold(
              appBar: appBar,
              body: const Center(
                child: CuText.bold14('Hello world!'),
              ),
            );
          },
        ),
      ];
}

import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
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
          name: 'Molecules/TextLoading',
          builder: (context) => CuTextLoading.med14(
            context.knobs.text(
              label: 'Text',
              initial: 'Loading',
            ),
          ),
        ),
        Story(
          name: 'Molecules/ScrollableHorizontalWrapper',
          builder: (context) => CuScrollableHorizontalWrapper(
            child: Placeholder(
              fallbackHeight: context.knobs.slider(
                label: 'Height',
                initial: 10,
                max: 100,
                min: 0,
              ),
              fallbackWidth: context.knobs.slider(
                label: 'Width',
                initial: 100,
                max: 1000,
                min: 10,
              ),
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
        Story(
          name: 'Molecules/SettingTile',
          builder: (context) {
            final titleText = context.knobs.text(
              label: 'Title',
              initial: 'IP Address',
            );

            final trailingText = context.knobs.text(
              label: 'Trailing',
              initial: '192.168.1.123',
            );

            return CuSettingTile(
              title: CuText.med14(titleText),
              trailing: CuText.med14(trailingText),
              onTap: () {},
            );
          },
        ),
        Story(
          name: 'Molecules/AlertDialog',
          builder: (context) {
            final titleText = context.knobs.text(
              label: 'Title',
              initial: 'Title',
            );

            final contentText = context.knobs.text(
              label: 'Content',
              initial: 'Content',
            );

            return CuAlertDialog(
              title: CuText.bold14(titleText),
              content: CuText.med14(contentText),
              actions: [
                CuElevatedButton(
                  text: 'OK',
                  onPressed: () {},
                ),
              ],
            );
          },
        ),
      ];
}

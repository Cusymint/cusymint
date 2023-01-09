import 'dart:math';

import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class CommonOrganisms extends StorybookPart {
  const CommonOrganisms();

  @override
  List<Story> get stories => [
        Story(
          name: 'Organisms/AnimatedHomeCard',
          builder: (context) {
            final hasAllCallbacks = context.knobs.boolean(
              label: 'Has all callbacks',
              initial: true,
            );

            final hasErrors = context.knobs.boolean(
              label: 'Has errors',
              initial: false,
            );

            final isLoading = context.knobs.boolean(
              label: 'Is loading',
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
              isLoading: isLoading,
              inputInTex: context.knobs.nullable.text(
                label: 'Input in TeX',
                initial: '\\int 15\\text{d}x',
              ),
              outputInTex: context.knobs.nullable.text(
                label: 'Output in TeX',
                initial: '7.5x^2 + C',
              ),
              steps: context.knobs.nullable.text(
                label: 'Steps',
                initial: '\\quad \\text{Simplify}: \\newline'
                    ' \\int 15\\text{d} x\\newline\\quad'
                    ' \\text{Solve integral:} \\int 15'
                    ' \\text{d} x = 15x + C:\\newline=\\qquad 15x',
              ),
              buttonRowCallbacks: hasAllCallbacks
                  ? CuButtonRowCallbacks(
                      onCopyTex: () {},
                      onCopyUtf: () {},
                      onShareUtf: () {},
                      onStepsTap: () {},
                    )
                  : null,
            );
          },
        ),
        Story(
          name: 'Organisms/ExampleIntegralsRow',
          builder: (context) {
            final texCards = <CuTexCard>[];

            for (var i = 0;
                i <
                    context.knobs.sliderInt(
                        label: 'Cards Count', initial: 3, min: 0, max: 15);
                i++) {
              texCards.add(
                CuTexCard(
                  '\\int 15sin(x) + \\frac{cos(x)}{sin(x) + 1}\\text{d}x',
                  onTap: () {},
                ),
              );
            }

            return CuExampleIntegralsRow(
              texCards: texCards,
            );
          },
        ),
        Story(
          name: 'Organisms/Drawer',
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
          name: 'Organisms/AboutWidget',
          builder: (context) => CuAboutWidget(onGithubTap: () {}),
        ),
        Story(
          name: 'Organisms/AuthorsWidget',
          builder: (context) => const CuAuthorsWidget(),
        ),
        Story(
          name: 'Organisms/SettingsList',
          builder: (context) {
            return CuSettingsList(
              settingTiles: [
                CuSettingTile(
                  onTap: () {},
                  title: const CuText('Language'),
                  trailing: const CuText('English'),
                ),
                CuSettingTile(
                  onTap: () {},
                  title: const CuText('IP Address'),
                  trailing: const CuText('localhost'),
                ),
                CuSettingTile(
                  onTap: () {},
                  title: const CuText('Licenses'),
                ),
              ],
            );
          },
        ),
        Story(
          name: 'Organisms/HistoryDialog',
          builder: (context) {
            final random = Random(123);
            final exampleInputs = [
              'int 15sin(x) + cos(x) dx',
              'int 15x + 5 dx',
              'int 15x^2 + 5x + 1 dx',
              'e^x+ 5x + 1',
            ];
            final itemsCount = context.knobs.sliderInt(
              label: 'Items count',
              initial: 4,
              min: 0,
              max: 40,
            );

            final historyItems = <CuHistoryItem>[
              for (var i = 0; i < itemsCount; ++i)
                CuHistoryItem(
                  exampleInputs[random.nextInt(exampleInputs.length)],
                  onTap: () {},
                )
            ];

            return CuHistoryDialog(
              historyItems: historyItems,
              onCancelPressed: () {},
              onClearHistoryPressed: () {},
            );
          },
        ),
        Story(
          name: 'Organisms/TextFieldAlertDialog',
          builder: (context) {
            final title = context.knobs.text(
              label: 'Title',
              initial: 'Title',
            );

            final isEnabled = context.knobs.boolean(
              label: 'Is enabled',
              initial: true,
            );

            return CuTextFieldAlertDialog(
              title: title,
              onCancelPressed: () {},
              onOkPressed: isEnabled ? () {} : null,
              textField: const CuTextField(),
            );
          },
        )
      ];
}

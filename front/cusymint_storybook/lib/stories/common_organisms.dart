import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class CommonOrganisms extends StorybookPart {
  const CommonOrganisms();

  @override
  List<Story> get stories => [
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
      ];
}

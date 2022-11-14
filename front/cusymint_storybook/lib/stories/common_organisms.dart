import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
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
      ];
}

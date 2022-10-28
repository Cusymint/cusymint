import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class TemplatesStories extends StorybookPart {
  const TemplatesStories();

  @override
  List<Story> get stories => [
        Story(
          name: 'Templates/WelcomePageTemplate',
          builder: (context) => CuWelcomePageTemplate(
            onTextFieldTap: () {},
          ),
        ),
        Story(
          name: 'Templates/SettingsPageTemplate',
          builder: (context) {
            final drawer = CuDrawer(
              onHomePressed: () {},
              onSettingsPressed: () {},
              onAboutPressed: () {},
            );

            return SettingsPageTemplate(
              drawer: drawer,
              chosenLanguage: 'English',
              ipAddress: 'localhost',
              onIpAddressTap: () {},
              onLanguageTap: () {},
              onLicensesTap: () {},
            );
          },
        ),
      ];
}

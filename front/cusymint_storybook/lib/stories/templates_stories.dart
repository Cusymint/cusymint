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
            return SettingsPageTemplate<String>(
              languageMenuItems: const [],
              chosenLanguage: 'English',
              ipAddress: 'localhost',
              onIpAddressTap: () {},
              onLicensesTap: () {},
            );
          },
        ),
        Story(
          name: 'Templates/AboutPageTemplate',
          builder: (context) => AboutTemplate(onGithubTap: () {}),
        ),
      ];
}

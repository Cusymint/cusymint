import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class CommonAtoms extends StorybookPart {
  const CommonAtoms();

  @override
  List<Story> get stories => [
        Story(
          name: 'Atoms/TextField',
          builder: (context) => const CuTextField(),
        ),
      ];
}

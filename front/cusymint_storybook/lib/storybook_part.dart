import 'package:storybook_flutter/storybook_flutter.dart';

abstract class StorybookPart {
  const StorybookPart();

  List<Story> get stories;
}

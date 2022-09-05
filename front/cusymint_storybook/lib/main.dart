import 'package:cusymint_storybook/stories/stories.dart';
import 'package:flutter/material.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

void main() {
  runApp(CusymintStorybook());
}

class CusymintStorybook extends StatelessWidget {
  CusymintStorybook({super.key});

  final _plugins = initializePlugins(
    contentsSidePanel: true,
    knobsSidePanel: true,
  );

  @override
  Widget build(BuildContext context) {
    return Storybook(
      plugins: _plugins,
      stories: [
        ...const CommonAtoms().stories,
      ],
    );
  }
}

import 'package:cusymint_storybook/stories/stories.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
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

  Widget _cusymintWrapper(context, child) => MaterialApp(
        title: 'cusymint',
        debugShowCheckedModeBanner: false,
        useInheritedMediaQuery: true,
        theme: CuTheme.of(context),
        // no dark theme exists for cusymint
        darkTheme: CuTheme.of(context),
        home: Scaffold(body: Center(child: child)),
      );

  @override
  Widget build(BuildContext context) {
    return Storybook(
      wrapperBuilder: _cusymintWrapper,
      plugins: _plugins,
      stories: [
        ...const CommonAtoms().stories,
        ...const CommonMolecules().stories,
      ],
    );
  }
}

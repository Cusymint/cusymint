import 'package:cusymint_storybook/stories/stories.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:storybook_flutter/storybook_flutter.dart';
// ignore: depend_on_referenced_packages
import 'package:easy_localization/easy_localization.dart';

void main() async {
  await CuL10n.ensureInitialized();

  runApp(CuL10n(child: CusymintStorybook()));
}

class CusymintStorybook extends StatelessWidget {
  CusymintStorybook({super.key});

  final _plugins = initializePlugins(
    contentsSidePanel: true,
    knobsSidePanel: true,
  );

  Widget _cusymintWrapper(BuildContext context, Widget? child) => MaterialApp(
        title: 'cusymint',
        debugShowCheckedModeBanner: false,
        useInheritedMediaQuery: true,
        theme: CuTheme.of(context),
        // no dark theme exists for cusymint
        darkTheme: CuTheme.of(context),
        locale: context.locale,
        supportedLocales: context.supportedLocales,
        localizationsDelegates: context.localizationDelegates,
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
        ...const CommonOrganisms().stories,
        ...const MainFlowStories().stories,
        ...const TemplatesStories().stories,
      ],
    );
  }
}

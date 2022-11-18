import 'package:auto_route/auto_route.dart';
import 'package:cusymint_app/features/about/pages/about_page.dart';
import 'package:cusymint_app/features/home/pages/home_page.dart';
import 'package:cusymint_app/features/home/pages/welcome_page.dart';
import 'package:cusymint_app/features/settings/pages/settings_page.dart';

@MaterialAutoRouter(
  replaceInRouteName: 'Page,Route',
  routes: <AutoRoute>[
    AutoRoute(page: WelcomePage, initial: true),
    AutoRoute(page: HomePage),
    AutoRoute(page: SettingsPage),
    AutoRoute(page: AboutPage),
  ],
)
class $AppRouter {}

// **************************************************************************
// AutoRouteGenerator
// **************************************************************************

// GENERATED CODE - DO NOT MODIFY BY HAND

// **************************************************************************
// AutoRouteGenerator
// **************************************************************************
//
// ignore_for_file: type=lint

// ignore_for_file: no_leading_underscores_for_library_prefixes
import 'package:auto_route/auto_route.dart' as _i5;
import 'package:cusymint_ui/cusymint_ui.dart' as _i7;
import 'package:flutter/material.dart' as _i6;

import '../about/pages/about_page.dart' as _i4;
import '../home/pages/home_page.dart' as _i2;
import '../home/pages/welcome_page.dart' as _i1;
import '../settings/pages/settings_page.dart' as _i3;

class AppRouter extends _i5.RootStackRouter {
  AppRouter([_i6.GlobalKey<_i6.NavigatorState>? navigatorKey])
      : super(navigatorKey);

  @override
  final Map<String, _i5.PageFactory> pagesMap = {
    WelcomeRoute.name: (routeData) {
      return _i5.MaterialPageX<dynamic>(
        routeData: routeData,
        child: const _i1.WelcomePage(),
      );
    },
    HomeRoute.name: (routeData) {
      final args =
          routeData.argsAs<HomeRouteArgs>(orElse: () => const HomeRouteArgs());
      return _i5.MaterialPageX<dynamic>(
        routeData: routeData,
        child: _i2.HomePage(
          key: args.key,
          isTextSelected: args.isTextSelected,
          initialExpression: args.initialExpression,
        ),
      );
    },
    SettingsRoute.name: (routeData) {
      return _i5.MaterialPageX<dynamic>(
        routeData: routeData,
        child: const _i3.SettingsPage(),
      );
    },
    AboutRoute.name: (routeData) {
      final args = routeData.argsAs<AboutRouteArgs>(
          orElse: () => const AboutRouteArgs());
      return _i5.MaterialPageX<dynamic>(
        routeData: routeData,
        child: _i4.AboutPage(key: args.key),
      );
    },
  };

  @override
  List<_i5.RouteConfig> get routes => [
        _i5.RouteConfig(
          WelcomeRoute.name,
          path: '/',
        ),
        _i5.RouteConfig(
          HomeRoute.name,
          path: '/home-page',
        ),
        _i5.RouteConfig(
          SettingsRoute.name,
          path: '/settings-page',
        ),
        _i5.RouteConfig(
          AboutRoute.name,
          path: '/about-page',
        ),
      ];
}

/// generated route for
/// [_i1.WelcomePage]
class WelcomeRoute extends _i5.PageRouteInfo<void> {
  const WelcomeRoute()
      : super(
          WelcomeRoute.name,
          path: '/',
        );

  static const String name = 'WelcomeRoute';
}

/// generated route for
/// [_i2.HomePage]
class HomeRoute extends _i5.PageRouteInfo<HomeRouteArgs> {
  HomeRoute({
    _i7.Key? key,
    bool isTextSelected = false,
    String? initialExpression,
  }) : super(
          HomeRoute.name,
          path: '/home-page',
          args: HomeRouteArgs(
            key: key,
            isTextSelected: isTextSelected,
            initialExpression: initialExpression,
          ),
        );

  static const String name = 'HomeRoute';
}

class HomeRouteArgs {
  const HomeRouteArgs({
    this.key,
    this.isTextSelected = false,
    this.initialExpression,
  });

  final _i7.Key? key;

  final bool isTextSelected;

  final String? initialExpression;

  @override
  String toString() {
    return 'HomeRouteArgs{key: $key, isTextSelected: $isTextSelected, initialExpression: $initialExpression}';
  }
}

/// generated route for
/// [_i3.SettingsPage]
class SettingsRoute extends _i5.PageRouteInfo<void> {
  const SettingsRoute()
      : super(
          SettingsRoute.name,
          path: '/settings-page',
        );

  static const String name = 'SettingsRoute';
}

/// generated route for
/// [_i4.AboutPage]
class AboutRoute extends _i5.PageRouteInfo<AboutRouteArgs> {
  AboutRoute({_i7.Key? key})
      : super(
          AboutRoute.name,
          path: '/about-page',
          args: AboutRouteArgs(key: key),
        );

  static const String name = 'AboutRoute';
}

class AboutRouteArgs {
  const AboutRouteArgs({this.key});

  final _i7.Key? key;

  @override
  String toString() {
    return 'AboutRouteArgs{key: $key}';
  }
}

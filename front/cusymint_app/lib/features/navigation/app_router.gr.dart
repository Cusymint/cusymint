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
import 'package:auto_route/auto_route.dart' as _i4;
import 'package:cusymint_ui/cusymint_ui.dart' as _i6;
import 'package:flutter/material.dart' as _i5;

import '../home/pages/home_page.dart' as _i2;
import '../home/pages/welcome_page.dart' as _i1;
import '../settings/pages/settings_page.dart' as _i3;

class AppRouter extends _i4.RootStackRouter {
  AppRouter([_i5.GlobalKey<_i5.NavigatorState>? navigatorKey])
      : super(navigatorKey);

  @override
  final Map<String, _i4.PageFactory> pagesMap = {
    WelcomeRoute.name: (routeData) {
      return _i4.MaterialPageX<dynamic>(
        routeData: routeData,
        child: const _i1.WelcomePage(),
      );
    },
    HomeRoute.name: (routeData) {
      final args =
          routeData.argsAs<HomeRouteArgs>(orElse: () => const HomeRouteArgs());
      return _i4.MaterialPageX<dynamic>(
        routeData: routeData,
        child: _i2.HomePage(
          key: args.key,
          isTextSelected: args.isTextSelected,
        ),
      );
    },
    SettingsRoute.name: (routeData) {
      return _i4.MaterialPageX<dynamic>(
        routeData: routeData,
        child: const _i3.SettingsPage(),
      );
    },
  };

  @override
  List<_i4.RouteConfig> get routes => [
        _i4.RouteConfig(
          WelcomeRoute.name,
          path: '/',
        ),
        _i4.RouteConfig(
          HomeRoute.name,
          path: '/home-page',
        ),
        _i4.RouteConfig(
          SettingsRoute.name,
          path: '/settings-page',
        ),
      ];
}

/// generated route for
/// [_i1.WelcomePage]
class WelcomeRoute extends _i4.PageRouteInfo<void> {
  const WelcomeRoute()
      : super(
          WelcomeRoute.name,
          path: '/',
        );

  static const String name = 'WelcomeRoute';
}

/// generated route for
/// [_i2.HomePage]
class HomeRoute extends _i4.PageRouteInfo<HomeRouteArgs> {
  HomeRoute({
    _i6.Key? key,
    bool isTextSelected = false,
  }) : super(
          HomeRoute.name,
          path: '/home-page',
          args: HomeRouteArgs(
            key: key,
            isTextSelected: isTextSelected,
          ),
        );

  static const String name = 'HomeRoute';
}

class HomeRouteArgs {
  const HomeRouteArgs({
    this.key,
    this.isTextSelected = false,
  });

  final _i6.Key? key;

  final bool isTextSelected;

  @override
  String toString() {
    return 'HomeRouteArgs{key: $key, isTextSelected: $isTextSelected}';
  }
}

/// generated route for
/// [_i3.SettingsPage]
class SettingsRoute extends _i4.PageRouteInfo<void> {
  const SettingsRoute()
      : super(
          SettingsRoute.name,
          path: '/settings-page',
        );

  static const String name = 'SettingsRoute';
}

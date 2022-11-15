/// GENERATED CODE - DO NOT MODIFY BY HAND
/// *****************************************************
///  FlutterGen
/// *****************************************************

// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: directives_ordering,unnecessary_import,implicit_dynamic_list_literal

import 'package:flutter/widgets.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter/services.dart';

class $AssetsFontsGen {
  const $AssetsFontsGen();

  /// File path: assets/fonts/MavenPro-Black.ttf
  String get mavenProBlack => 'assets/fonts/MavenPro-Black.ttf';

  /// File path: assets/fonts/MavenPro-Bold.ttf
  String get mavenProBold => 'assets/fonts/MavenPro-Bold.ttf';

  /// File path: assets/fonts/MavenPro-ExtraBold.ttf
  String get mavenProExtraBold => 'assets/fonts/MavenPro-ExtraBold.ttf';

  /// File path: assets/fonts/MavenPro-Medium.ttf
  String get mavenProMedium => 'assets/fonts/MavenPro-Medium.ttf';

  /// File path: assets/fonts/MavenPro-Regular.ttf
  String get mavenProRegular => 'assets/fonts/MavenPro-Regular.ttf';

  /// File path: assets/fonts/MavenPro-SemiBold.ttf
  String get mavenProSemiBold => 'assets/fonts/MavenPro-SemiBold.ttf';

  /// List of all assets
  List<String> get values => [
        mavenProBlack,
        mavenProBold,
        mavenProExtraBold,
        mavenProMedium,
        mavenProRegular,
        mavenProSemiBold
      ];
}

class $AssetsIconsGen {
  const $AssetsIconsGen();

  /// File path: assets/icons/app_icon_background.png
  AssetGenImage get appIconBackground =>
      const AssetGenImage('assets/icons/app_icon_background.png');

  /// File path: assets/icons/app_icon_curved_edges.png
  AssetGenImage get appIconCurvedEdges =>
      const AssetGenImage('assets/icons/app_icon_curved_edges.png');

  /// File path: assets/icons/app_icon_foreground.png
  AssetGenImage get appIconForeground =>
      const AssetGenImage('assets/icons/app_icon_foreground.png');

  /// File path: assets/icons/app_icon_ios.png
  AssetGenImage get appIconIos =>
      const AssetGenImage('assets/icons/app_icon_ios.png');

  /// File path: assets/icons/copy.svg
  SvgGenImage get copy => const SvgGenImage('assets/icons/copy.svg');

  /// File path: assets/icons/copy_tex.svg
  SvgGenImage get copyTex => const SvgGenImage('assets/icons/copy_tex.svg');

  /// File path: assets/icons/share.svg
  SvgGenImage get share => const SvgGenImage('assets/icons/share.svg');

  /// List of all assets
  List<dynamic> get values => [
        appIconBackground,
        appIconCurvedEdges,
        appIconForeground,
        appIconIos,
        copy,
        copyTex,
        share
      ];
}

class $AssetsImagesGen {
  const $AssetsImagesGen();

  /// File path: assets/images/drawer_background.jpg
  AssetGenImage get drawerBackground =>
      const AssetGenImage('assets/images/drawer_background.jpg');

  /// File path: assets/images/logo_wide.png
  AssetGenImage get logoWide =>
      const AssetGenImage('assets/images/logo_wide.png');

  /// List of all assets
  List<AssetGenImage> get values => [drawerBackground, logoWide];
}

class $AssetsSvgGen {
  const $AssetsSvgGen();

  /// File path: assets/svg/integral_icon.svg
  SvgGenImage get integralIcon =>
      const SvgGenImage('assets/svg/integral_icon.svg');

  /// File path: assets/svg/logo_wide.svg
  SvgGenImage get logoWide => const SvgGenImage('assets/svg/logo_wide.svg');

  /// List of all assets
  List<SvgGenImage> get values => [integralIcon, logoWide];
}

class CuAssets {
  CuAssets._();

  static const $AssetsFontsGen fonts = $AssetsFontsGen();
  static const $AssetsIconsGen icons = $AssetsIconsGen();
  static const $AssetsImagesGen images = $AssetsImagesGen();
  static const $AssetsSvgGen svg = $AssetsSvgGen();
}

class AssetGenImage {
  const AssetGenImage(this._assetName);

  final String _assetName;

  Image image({
    Key? key,
    AssetBundle? bundle,
    ImageFrameBuilder? frameBuilder,
    ImageErrorWidgetBuilder? errorBuilder,
    String? semanticLabel,
    bool excludeFromSemantics = false,
    double? scale,
    double? width,
    double? height,
    Color? color,
    Animation<double>? opacity,
    BlendMode? colorBlendMode,
    BoxFit? fit,
    AlignmentGeometry alignment = Alignment.center,
    ImageRepeat repeat = ImageRepeat.noRepeat,
    Rect? centerSlice,
    bool matchTextDirection = false,
    bool gaplessPlayback = false,
    bool isAntiAlias = false,
    String? package = 'cusymint_assets',
    FilterQuality filterQuality = FilterQuality.low,
    int? cacheWidth,
    int? cacheHeight,
  }) {
    return Image.asset(
      _assetName,
      key: key,
      bundle: bundle,
      frameBuilder: frameBuilder,
      errorBuilder: errorBuilder,
      semanticLabel: semanticLabel,
      excludeFromSemantics: excludeFromSemantics,
      scale: scale,
      width: width,
      height: height,
      color: color,
      opacity: opacity,
      colorBlendMode: colorBlendMode,
      fit: fit,
      alignment: alignment,
      repeat: repeat,
      centerSlice: centerSlice,
      matchTextDirection: matchTextDirection,
      gaplessPlayback: gaplessPlayback,
      isAntiAlias: isAntiAlias,
      package: package,
      filterQuality: filterQuality,
      cacheWidth: cacheWidth,
      cacheHeight: cacheHeight,
    );
  }

  ImageProvider provider() => AssetImage(_assetName);

  String get path => _assetName;

  String get keyName => 'packages/cusymint_assets/$_assetName';
}

class SvgGenImage {
  const SvgGenImage(this._assetName);

  final String _assetName;

  SvgPicture svg({
    Key? key,
    bool matchTextDirection = false,
    AssetBundle? bundle,
    String? package = 'cusymint_assets',
    double? width,
    double? height,
    BoxFit fit = BoxFit.contain,
    AlignmentGeometry alignment = Alignment.center,
    bool allowDrawingOutsideViewBox = false,
    WidgetBuilder? placeholderBuilder,
    Color? color,
    BlendMode colorBlendMode = BlendMode.srcIn,
    String? semanticsLabel,
    bool excludeFromSemantics = false,
    Clip clipBehavior = Clip.hardEdge,
    bool cacheColorFilter = false,
    SvgTheme? theme,
  }) {
    return SvgPicture.asset(
      _assetName,
      key: key,
      matchTextDirection: matchTextDirection,
      bundle: bundle,
      package: package,
      width: width,
      height: height,
      fit: fit,
      alignment: alignment,
      allowDrawingOutsideViewBox: allowDrawingOutsideViewBox,
      placeholderBuilder: placeholderBuilder,
      color: color,
      colorBlendMode: colorBlendMode,
      semanticsLabel: semanticsLabel,
      excludeFromSemantics: excludeFromSemantics,
      clipBehavior: clipBehavior,
      cacheColorFilter: cacheColorFilter,
      theme: theme,
    );
  }

  String get path => 'packages/cusymint_assets/$_assetName';
}

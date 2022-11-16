import 'package:cusymint_assets/cusymint_assets.dart';
import 'package:flutter/material.dart';

class CuBlackBoardDecoration extends BoxDecoration {
  CuBlackBoardDecoration()
      : super(
          image: DecorationImage(
            // `CuAssets.images.drawerBackground.provider()` doesn't work here
            // due to bug in the package, because it doesn't prefix path with
            // `packages/cusymint_assets/`.
            image: AssetImage(CuAssets.images.drawerBackground.keyName),
            fit: BoxFit.cover,
          ),
        );
}

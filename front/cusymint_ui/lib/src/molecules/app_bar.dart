import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuAppBar extends AppBar {
  CuAppBar({
    this.onMenuPressed,
    super.key,
  }) : super(
          // TODO: Fix sizing, logo is larger than IconButton
          title: const CuLogo(),
          centerTitle: true,
          actions: [
            if (onMenuPressed != null)
              IconButton(
                icon: const Icon(Icons.menu),
                onPressed: onMenuPressed,
              ),
          ],
          elevation: 0,
          backgroundColor: Colors.transparent,
        );

  final VoidCallback? onMenuPressed;
}
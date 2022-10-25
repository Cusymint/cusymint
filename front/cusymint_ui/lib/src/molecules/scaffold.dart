import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuScaffold extends StatelessWidget {
  const CuScaffold({
    super.key,
    required this.body,
    this.drawer,
    this.appBar,
  });

  final Widget body;

  final CuAppBar? appBar;
  final CuDrawer? drawer;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: appBar,
      backgroundColor: CuColors.of(context).mint,
      body: body,
      drawer: drawer,
    );
  }
}

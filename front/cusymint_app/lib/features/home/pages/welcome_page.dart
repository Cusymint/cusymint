import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class WelcomePage extends StatelessWidget {
  const WelcomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        backgroundColor: CuColors.of(context).mint,
      ),
    );
  }
}

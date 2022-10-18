import 'package:auto_route/auto_route.dart';
import 'package:cusymint_app/features/navigation/app_router.gr.dart';
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

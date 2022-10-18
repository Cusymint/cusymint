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
        body: Column(
          children: [
            Expanded(
              flex: 2,
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Hero(tag: 'logo', child: CuLogo()),
                    CuText.bold14(
                      'Cuda symbolic integration',
                      color: CuColors.of(context).white,
                    ),
                  ],
                ),
              ),
            ),
            Expanded(
              flex: 3,
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Column(
                  children: [
                    SizedBox(
                      width: 400,
                      child: Hero(
                        tag: 'input',
                        child: CuTextField(
                          onTap: () {
                            context.router.push(const HomeRoute());
                          },
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

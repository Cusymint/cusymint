import 'dart:math';

import 'package:cusymint_app/features/home/models/example_integral.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class ExampleIntegralsCubit extends Cubit<ExampleIntegralsState> {
  ExampleIntegralsCubit({Random? random})
      : _random = random ?? Random(),
        super(const ExampleIntegralsState(integrals: [])) {
    generateIntegrals();
  }

  final Random _random;
  final List<ExampleIntegral> exampleIntegrals = [
    const ExampleIntegral(
      inputInTex: '\\int x^2 - 53 + 2x^6 \\, \\text{d}x',
      inputInUtf: 'x^2 - 53 + 2*x^6',
    ),
    const ExampleIntegral(
      inputInTex: '\\int e^x \\cdot e^{e^x} \\cdot e^{e^{e^x}} \\, \\text{d}x',
      inputInUtf: 'e^x * e^(e^x) * e^(e^(e^x))',
    ),
    const ExampleIntegral(
      inputInTex: '\\int \\sin^2(x) + \\cos^2(x) \\, \\text{d}x',
      inputInUtf: '(sin(x))^2 + (cos(x))^2',
    ),
    const ExampleIntegral(
      inputInTex: '\\int \\frac{1}{x} \\, \\text{d}x',
      inputInUtf: '1/x',
    ),
    const ExampleIntegral(
      inputInTex: '\\int ax^2 + bx + c \\, \\text{d}x',
      inputInUtf: 'a*x^2 + b*x + c',
    ),
  ];

  /// Generates a list of example integrals.
  ///
  /// List count is limited by number of available example integrals
  /// or [integralCount], whichever is smaller.
  void generateIntegrals({
    int integralCount = 5,
  }) {
    exampleIntegrals.shuffle(_random);

    final integrals = exampleIntegrals.take(integralCount).toList();

    emit(ExampleIntegralsState(integrals: integrals));
  }
}

class ExampleIntegralsState {
  const ExampleIntegralsState({
    required this.integrals,
  });

  final List<ExampleIntegral> integrals;
  int get length => integrals.length;
}

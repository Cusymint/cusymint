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
      inputInTex: '\\int x^2 - 53 + 2*x^6 \\, \text{d}x',
      inputInUtf: 'x^2 - 53 + 2*x^6',
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

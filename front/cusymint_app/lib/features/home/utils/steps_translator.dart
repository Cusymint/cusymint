import 'package:cusymint_l10n/cusymint_l10n.dart';

abstract class StepsTranslator {
  static String? translate(String? steps) {
    if (steps == null) {
      return null;
    }

    return _translateSteps(steps);
  }

  static String _translateSteps(String steps) {
    _translatableValues.forEach((key, translate) {
      steps = steps.replaceAll('[[$key]]', translate());
    });

    return steps;
  }

  static final Map<String, String Function()> _translatableValues = {
    'substitute': () => Strings.stepSubstitute.tr(),
    'splitSum': () => Strings.stepSplitSum.tr(),
    'integrateByParts': () => Strings.stepIntegrateByParts.tr(),
    'solveIntegral': () => Strings.stepSolveIntegral.tr(),
    'bringOutConstant': () => Strings.stepBringOutConstant.tr(),
    'simplify': () => Strings.stepSimplify.tr(),
    'integral': () => Strings.stepIntegral.tr(),
    'stSuffix': () => Strings.stepStSuffix.tr(),
    'ndSuffix': () => Strings.stepNdSuffix.tr(),
    'rdSuffix': () => Strings.stepRdSuffix.tr(),
    'thSuffix': () => Strings.stepThSuffix.tr(),
  };
}

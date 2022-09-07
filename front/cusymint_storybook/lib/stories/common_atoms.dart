import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class CommonAtoms extends StorybookPart {
  const CommonAtoms();

  @override
  List<Story> get stories => [
        Story(
          name: 'Atoms/TextField',
          builder: (context) => CuTextField(
            prefixIcon: context.knobs.nullable.options(
              label: 'PrefixIcon',
              initial: const Icon(Icons.ac_unit),
              options: [],
            ),
            suffixIcon: context.knobs.nullable.options(
              label: 'SuffixIcon',
              initial: const Icon(
                Icons.close,
              ),
            ),
          ),
        ),
        Story(
          name: 'Atoms/Card',
          builder: (context) => const CuCard(),
        ),
      ];
}
import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class CommonAtoms extends StorybookPart {
  const CommonAtoms();

  @override
  List<Story> get stories => [
        Story(
          name: 'Atoms/Card',
          builder: (context) => const SizedBox(
            width: 150,
            height: 100,
            child: CuCard(
              child: Center(
                child: CuText('Hello world!'),
              ),
            ),
          ),
        ),
        Story(
          name: 'Atoms/Logo',
          builder: (context) => CuLogo(
            color: context.knobs.options(
              label: 'Color',
              initial: Colors.black,
              options: [
                const Option(label: 'Black', value: Colors.black),
                const Option(label: 'White', value: Colors.white),
              ],
            ),
          ),
        ),
        Story(
          name: 'Atoms/TextField',
          builder: (context) => SizedBox(
            width: 350,
            child: CuTextField(
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
        ),
        Story(
          name: 'Atoms/Text',
          builder: (context) => CuText(
            context.knobs.text(
              label: 'Text',
              initial: 'Hello world!',
            ),
          ),
        ),
        Story(
          name: 'Atoms/Toast',
          builder: (context) => CuToast(
            message: context.knobs.text(
              label: 'Message',
              initial: 'Hello world!',
            ),
          ),
        ),
      ];
}

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
                child: CuText.bold14('Hello world!'),
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
            child: Theme(
              data: CuTheme.of(context),
              child: Scaffold(
                body: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
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
              ),
            ),
          ),
        ),
        Story(
          name: 'Atoms/Text',
          builder: (context) {
            final text = context.knobs.text(
              label: 'Text',
              initial: 'Hello world!',
            );

            return context.knobs.options(
              label: 'Type',
              initial: CuText(text),
              options: [
                Option(label: 'medium 14', value: CuText.med14(text)),
                Option(label: 'bold 14', value: CuText.bold14(text)),
                Option(label: 'medium 24', value: CuText.med24(text)),
              ],
            );
          },
        ),
        Story(
          name: 'Atoms/Toast',
          builder: (context) {
            final bool isError = context.knobs.boolean(
              label: 'Is error',
              initial: false,
            );
            final String message = context.knobs.text(
              label: 'Message',
              initial: 'Hello world!',
            );
            return isError
                ? CuToast.error(message: message)
                : CuToast.success(message: message);
          },
        ),
        Story(
            name: 'Atoms/ElevatedButton',
            builder: (context) {
              final text = context.knobs.text(
                label: 'Text',
                initial: 'Solve',
              );

              return CuElevatedButton(
                text: text,
                onPressed: () {},
              );
            })
      ];
}

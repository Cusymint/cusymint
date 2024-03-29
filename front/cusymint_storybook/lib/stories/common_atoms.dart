import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
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
            name: 'Atoms/CopyTexIconButton',
            builder: (context) {
              return Row(
                children: [
                  IconButton(onPressed: () {}, icon: Icon(Icons.copy)),
                  CuCopyTexIconButton(
                    onPressed: () {},
                  ),
                ],
              );
            }),
        Story(
          name: 'Atoms/Logo',
          builder: (context) => CuLogo(
            color: context.knobs.options(
              label: 'Color',
              initial: Colors.white,
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
          name: 'Atoms/ShimmerRectangle',
          builder: (context) {
            final width = context.knobs.slider(
              label: 'Width',
              min: 50,
              max: 500,
              initial: 200,
            );
            final height = context.knobs.slider(
              label: 'Height',
              min: 5,
              max: 500,
              initial: 20,
            );

            return CuShimmerRectangle(
              width: width,
              height: height,
            );
          },
        ),
        Story(
          name: 'Atoms/Buttons/ElevatedButton',
          builder: (context) {
            final text = context.knobs.text(
              label: 'Text',
              initial: 'Solve',
            );

            return CuElevatedButton(
              text: text,
              onPressed: () {},
            );
          },
        ),
        Story(
          name: 'Atoms/Buttons/TextButton',
          builder: (context) {
            final text = context.knobs.text(
              label: 'Text',
              initial: 'Cancel',
            );

            return CuTextButton(
              text: text,
              onPressed: () {},
            );
          },
        ),
      ];
}

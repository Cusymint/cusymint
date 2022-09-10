import 'package:cusymint_storybook/storybook_part.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';
import 'package:storybook_flutter/storybook_flutter.dart';

class CommonMolecules extends StorybookPart {
  const CommonMolecules();

  @override
  List<Story> get stories => [
        Story(
          name: 'Molecules/Carousel',
          builder: (context) => SizedBox(
            height: 100,
            child: CuCarousel(
              children: [
                for (int i = 0;
                    i <
                        context.knobs.sliderInt(
                          label: 'Children count',
                          initial: 5,
                        );
                    ++i)
                  CuCard(
                    child: CuText('Child #${i + 1}'),
                  ),
              ],
            ),
          ),
        ),
      ];
}

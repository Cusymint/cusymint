import 'package:cusymint_ui/cusymint_ui.dart';

class CuInterpretErrorCard extends StatelessWidget {
  const CuInterpretErrorCard({super.key, required this.errors});

  final List<String> errors;

  @override
  Widget build(BuildContext context) {
    return CuCard(
      child: Column(children: [
        for (final error in errors)
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: CuText.med14(error),
          ),
      ]),
    );
  }
}

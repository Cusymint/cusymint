import 'package:cusymint_ui/cusymint_ui.dart';

class CuInterpretLoadingCard extends StatelessWidget {
  const CuInterpretLoadingCard({super.key});

  @override
  Widget build(BuildContext context) {
    return CuCard(
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: const CuShimmerRectangle(width: 200, height: 20),
          ),
        ],
      ),
    );
  }
}

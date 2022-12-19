import 'package:cusymint_ui/cusymint_ui.dart';

class CuExampleIntegralsRow extends StatelessWidget {
  const CuExampleIntegralsRow({super.key, required this.texCards});

  final List<CuTexCard> texCards;

  @override
  Widget build(BuildContext context) {
    return CuScrollableHorizontalWrapper(
      thumbVisibility: false,
      child: Row(
        children: texCards,
      ),
    );
  }
}

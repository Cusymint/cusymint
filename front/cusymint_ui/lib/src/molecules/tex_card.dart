import 'package:cusymint_ui/cusymint_ui.dart';

class CuTexCard extends StatelessWidget {
  const CuTexCard(
    this.data, {
    super.key,
    required this.onTap,
  });

  final String data;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return CuCard(
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: TexView(data),
        ),
      ),
    );
  }
}

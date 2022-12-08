import 'package:cusymint_ui/cusymint_ui.dart';

class CuHistoryItem extends StatelessWidget {
  const CuHistoryItem(
    this.data, {
    super.key,
    required this.onTap,
  });

  final String data;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: CuText.med14(data),
      ),
    );
  }
}

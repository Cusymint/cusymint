import 'package:cusymint_ui/cusymint_ui.dart';

class CuSettingTile extends StatelessWidget {
  const CuSettingTile({
    super.key,
    this.onTap,
    required this.title,
    this.trailing,
  });

  final VoidCallback? onTap;
  final CuText title;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      onTap: onTap,
      title: title,
      trailing: trailing,
    );
  }
}

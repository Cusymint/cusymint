import 'package:cusymint_ui/cusymint_ui.dart';

class CuSettingsList extends StatelessWidget {
  const CuSettingsList({
    super.key,
    required this.settingTiles,
  });

  final List<Widget> settingTiles;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 400,
      child: ListView(children: settingTiles),
    );
  }
}

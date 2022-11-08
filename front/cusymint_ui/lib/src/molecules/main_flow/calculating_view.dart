import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';

class CuCalculatingView extends StatelessWidget {
  const CuCalculatingView({super.key});

  @override
  Widget build(BuildContext context) {
    return CuTextLoading.med14(Strings.calculating.tr());
  }
}

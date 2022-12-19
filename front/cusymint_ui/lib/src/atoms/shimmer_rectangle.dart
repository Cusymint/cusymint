import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:shimmer/shimmer.dart';

class CuShimmerRectangle extends StatelessWidget {
  const CuShimmerRectangle({
    super.key,
    this.width,
    this.height,
  });

  final double? width;
  final double? height;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return Shimmer.fromColors(
      baseColor: colors.grayDark,
      highlightColor: colors.gray,
      child: Container(
        decoration: BoxDecoration(
          color: colors.black,
          borderRadius: const BorderRadius.all(Radius.circular(8)),
        ),
        height: height,
        width: width,
      ),
    );
  }
}

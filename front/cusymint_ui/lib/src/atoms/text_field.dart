import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/foundation.dart';

class CuTextField extends StatelessWidget {
  const CuTextField({
    super.key,
    this.controller,
    this.onTap,
    this.onChanged,
    this.prefixIcon,
    this.suffixIcon,
    this.label,
    this.onSubmitted,
  });

  final TextEditingController? controller;
  final VoidCallback? onTap;
  final void Function(String newText)? onChanged;
  final void Function(String submittedText)? onSubmitted;
  final Widget? prefixIcon;
  final Widget? suffixIcon;
  final Widget? label;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    return Material(
      type: MaterialType.transparency,
      child: TextField(
        controller: controller,
        onTap: onTap,
        onChanged: onChanged,
        onSubmitted: onSubmitted,
        style: TextStyle(
          color: colors.black,
          fontSize: 18.0,
        ),
        decoration: InputDecoration(
          prefixIcon: prefixIcon,
          suffixIcon: suffixIcon,
          prefixIconColor: colors.mintDark,
          suffixIconColor: colors.mintDark,
          label: label,
          border: _CuInputBorder(color: colors.mintHeavyish),
          enabledBorder: _CuInputBorder(color: colors.mintHeavyish),
          focusedBorder: _CuInputBorder(color: colors.mintDark),
          errorBorder: _CuInputBorder(color: colors.errorColor),
          focusColor: colors.mintDark,
          filled: true,
          fillColor: colors.mintLight,
          iconColor: colors.mintDark,
        ),
      ),
    );
  }
}

class _CuInputBorder extends _DecoratedInputBorder {
  _CuInputBorder({required this.color})
      : super(
          child: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8.0),
            borderSide: BorderSide(color: color, width: 2.0),
          ),
          shadow: BoxShadow(
            offset: const Offset(0, 4),
            spreadRadius: 0,
            blurRadius: 4,
            color: Colors.black.withOpacity(0.25),
          ),
        );

  final CuColor color;
}

// https://stackoverflow.com/questions/54194347/how-to-add-drop-shadow-to-textformfield-in-flutter
class _DecoratedInputBorder extends InputBorder {
  _DecoratedInputBorder({
    required this.child,
    required this.shadow,
  }) : super(borderSide: child.borderSide);

  final InputBorder child;

  final BoxShadow shadow;

  @override
  bool get isOutline => child.isOutline;

  @override
  Path getInnerPath(Rect rect, {TextDirection? textDirection}) =>
      child.getInnerPath(rect, textDirection: textDirection);

  @override
  Path getOuterPath(Rect rect, {TextDirection? textDirection}) =>
      child.getOuterPath(rect, textDirection: textDirection);

  @override
  EdgeInsetsGeometry get dimensions => child.dimensions;

  @override
  InputBorder copyWith(
      {BorderSide? borderSide,
      InputBorder? child,
      BoxShadow? shadow,
      bool? isOutline}) {
    return _DecoratedInputBorder(
      child: (child ?? this.child).copyWith(borderSide: borderSide),
      shadow: shadow ?? this.shadow,
    );
  }

  @override
  ShapeBorder scale(double t) {
    final scaledChild = child.scale(t);

    return _DecoratedInputBorder(
      child: scaledChild is InputBorder ? scaledChild : child,
      shadow: BoxShadow.lerp(null, shadow, t)!,
    );
  }

  @override
  void paint(Canvas canvas, Rect rect,
      {double? gapStart,
      double gapExtent = 0.0,
      double gapPercentage = 0.0,
      TextDirection? textDirection}) {
    final clipPath = Path()
      ..addRect(const Rect.fromLTWH(-5000, -5000, 10000, 10000))
      ..addPath(getInnerPath(rect), Offset.zero)
      ..fillType = PathFillType.evenOdd;
    canvas.clipPath(clipPath);

    final Paint paint = shadow.toPaint();
    final Rect bounds = rect.shift(shadow.offset).inflate(shadow.spreadRadius);

    canvas.drawPath(getOuterPath(bounds), paint);

    child.paint(canvas, rect,
        gapStart: gapStart,
        gapExtent: gapExtent,
        gapPercentage: gapPercentage,
        textDirection: textDirection);
  }

  @override
  bool operator ==(Object other) {
    if (other.runtimeType != runtimeType) return false;
    return other is _DecoratedInputBorder &&
        other.borderSide == borderSide &&
        other.child == child &&
        other.shadow == shadow;
  }

  @override
  int get hashCode => Object.hash(borderSide, child, shadow);

  @override
  String toString() {
    return '${objectRuntimeType(this, 'DecoratedInputBorder')}($borderSide, $shadow, $child)';
  }
}

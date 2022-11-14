import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/gestures.dart';

class CuText extends StatelessWidget {
  const CuText(
    this.data, {
    super.key,
    this.color,
    this.fontWeight = FontWeight.normal,
    this.fontSize = 14.0,
  });

  const CuText.med14(
    this.data, {
    super.key,
    this.color,
  })  : fontWeight = FontWeight.normal,
        fontSize = 14.0;

  const CuText.bold14(
    this.data, {
    super.key,
    this.color,
  })  : fontWeight = FontWeight.bold,
        fontSize = 14.0;

  const CuText.med24(
    this.data, {
    super.key,
    this.color,
  })  : fontWeight = FontWeight.normal,
        fontSize = 24.0;

  final String data;
  final CuColor? color;
  final FontWeight fontWeight;
  final double fontSize;

  @override
  Widget build(BuildContext context) {
    final textColor = color ?? CuColors.of(context).textBlack;

    return Text(
      data,
      style: CuTextStyle(
        color: textColor,
        fontWeight: fontWeight,
        fontSize: fontSize,
      ),
    );
  }
}

class CuTextSpan extends TextSpan {
  const CuTextSpan({
    String? text,
    CuTextStyle? style,
    List<InlineSpan>? children,
  }) : super(
          text: text,
          style: style,
          children: children,
        );

  CuTextSpan.med14({
    String? text,
    CuColor? color,
    List<InlineSpan>? children,
  }) : super(
          text: text,
          style: CuTextStyle.med14(color: color),
          children: children,
        );

  CuTextSpan.link14({
    String? text,
    required VoidCallback onTap,
    CuColor? color,
    List<InlineSpan>? children,
  }) : super(
          text: text,
          style: CuTextStyle.med14(color: color),
          children: children,
          recognizer: TapGestureRecognizer()..onTap = onTap,
          mouseCursor: SystemMouseCursors.click,
        );
}

class CuTextStyle extends TextStyle {
  const CuTextStyle({
    CuColor? color,
    FontWeight fontWeight = FontWeight.normal,
    double fontSize = 14.0,
  }) : super(color: color, fontWeight: fontWeight, fontSize: fontSize);

  const CuTextStyle.med14({
    CuColor? color,
  }) : super(fontWeight: FontWeight.normal, fontSize: 14.0, color: color);

  const CuTextStyle.bold14({
    CuColor? color,
  }) : super(fontWeight: FontWeight.bold, fontSize: 14.0, color: color);

  const CuTextStyle.med24({
    CuColor? color,
  }) : super(fontWeight: FontWeight.normal, fontSize: 24.0, color: color);
}

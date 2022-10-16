import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuToast extends StatelessWidget {
  const CuToast({super.key, required this.message, this.color})
      : _type = _CuToastType.unknown;

  const CuToast.error({super.key, required this.message})
      : _type = _CuToastType.error,
        color = null;

  const CuToast.success({super.key, required this.message})
      : _type = _CuToastType.success,
        color = null;

  final String message;
  final CuColor? color;
  final _CuToastType _type;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: _getColor(context),
        borderRadius: BorderRadius.circular(8.0),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.25),
            spreadRadius: 0,
            blurRadius: 4,
            offset: const Offset(8, 8),
          ),
        ],
      ),
      child: CuText.bold14(message),
    );
  }

  CuColor _getColor(BuildContext context) {
    if (color != null) {
      return color!;
    }

    switch (_type) {
      case _CuToastType.error:
        return CuColors.of(context).errorColor;
      case _CuToastType.success:
        return CuColors.of(context).mintHeavyish;
      default:
        return CuColors.of(context).mint;
    }
  }
}

enum _CuToastType {
  unknown,
  error,
  success,
}

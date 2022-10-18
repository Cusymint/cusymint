import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuTextField extends StatelessWidget {
  const CuTextField({
    super.key,
    this.controller,
    this.onTap,
    this.onChanged,
    this.prefixIcon,
    this.suffixIcon,
    this.label,
  });

  final TextEditingController? controller;
  final VoidCallback? onTap;
  final void Function(String newText)? onChanged;
  final Widget? prefixIcon;
  final Widget? suffixIcon;
  final Widget? label;

  @override
  Widget build(BuildContext context) {
    final colors = CuColors.of(context);

    // TODO: fix icon colors
    // TODO: add shadow
    return TextField(
      controller: controller,
      onTap: onTap,
      onChanged: onChanged,
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
        border: _CuInputBorder(color: colors.mintDark),
        enabledBorder: _CuInputBorder(color: colors.mintDark),
        focusedBorder: _CuInputBorder(color: colors.mintHeavyish),
        errorBorder: _CuInputBorder(color: colors.errorColor),
        focusColor: colors.mintDark,
        filled: true,
        fillColor: colors.mintLight,
        iconColor: colors.mintDark,
      ),
    );
  }
}

class _CuInputBorder extends OutlineInputBorder {
  _CuInputBorder({
    required CuColor color,
  }) : super(
          borderRadius: BorderRadius.circular(8.0),
          borderSide: BorderSide(color: color, width: 2.0),
        );
}

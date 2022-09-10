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
    // TODO: apply app style
    return TextField(
      controller: controller,
      onTap: onTap,
      onChanged: onChanged,
      decoration: InputDecoration(
        prefixIcon: prefixIcon,
        suffixIcon: suffixIcon,
        border: const OutlineInputBorder(),
        label: label,
      ),
    );
  }
}

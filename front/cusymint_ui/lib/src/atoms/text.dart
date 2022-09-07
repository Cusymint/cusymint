import 'package:flutter/material.dart';

class CuText extends StatelessWidget {
  const CuText(
    this.data, {
    super.key,
  });

  final String data;

  @override
  Widget build(BuildContext context) {
    return Text(data);
  }
}

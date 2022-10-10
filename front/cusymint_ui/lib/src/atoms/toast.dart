import 'package:flutter/material.dart';

import 'atoms.dart';

class CuToast extends StatelessWidget {
  const CuToast({super.key, required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: Colors.greenAccent.withOpacity(0.8),
        borderRadius: BorderRadius.circular(12.0),
      ),
      child: CuText(message),
    );
  }
}

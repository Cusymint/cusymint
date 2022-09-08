import 'package:cusymint_app/features/home/pages/home_page.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(const CusymintApp());
}

class CusymintApp extends StatelessWidget {
  const CusymintApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: 'cusymint',
      home: HomePage(),
    );
  }
}

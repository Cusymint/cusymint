import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';

class CuTextLoading extends StatefulWidget {
  const CuTextLoading(
    this.data, {
    super.key,
    this.color,
    this.fontWeight = FontWeight.normal,
    this.fontSize = 14.0,
  });

  const CuTextLoading.med14(
    this.data, {
    super.key,
    this.color,
  })  : fontWeight = FontWeight.normal,
        fontSize = 14.0;

  final String data;
  final CuColor? color;
  final FontWeight fontWeight;
  final double fontSize;

  @override
  State<CuTextLoading> createState() => _CuTextLoadingState();
}

class _CuTextLoadingState extends State<CuTextLoading>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<int> _dotsAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    );

    _dotsAnimation = IntTween(
      begin: 1,
      end: 3,
    ).animate(_controller);

    _controller.repeat();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _dotsAnimation,
      builder: (context, child) => CuText(
        widget.data + '.' * _dotsAnimation.value,
        color: widget.color,
        fontWeight: widget.fontWeight,
        fontSize: widget.fontSize,
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}

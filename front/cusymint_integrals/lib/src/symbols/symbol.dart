abstract class Symbol {
  const Symbol();

  String toTex();
  String toUtf();

  @override
  String toString() {
    return toUtf();
  }
}

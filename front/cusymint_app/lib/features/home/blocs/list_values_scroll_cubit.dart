import 'package:flutter_bloc/flutter_bloc.dart';

class ListValuesScrollCubit<T> extends Cubit<ListValuesScrollState<T>> {
  ListValuesScrollCubit({
    required List<T> values,
    int startingIndex = 0,
  })  : assert(values.isEmpty ||
            (startingIndex >= 0 && startingIndex < values.length)),
        super(
          ListValuesScrollState(values: values, currentIndex: startingIndex),
        );

  void next() {
    if (state.values.isEmpty) return;

    if (state.hasNext) {
      emit(ListValuesScrollState(
        values: state.values,
        currentIndex: state.currentIndex + 1,
      ));
    }
  }

  void previous() {
    if (state.values.isEmpty) return;

    if (state.hasPrevious) {
      emit(ListValuesScrollState(
        values: state.values,
        currentIndex: state.currentIndex - 1,
      ));
    }
  }

  void nextOverlap() {
    if (state.values.isEmpty) return;

    emit(ListValuesScrollState(
      values: state.values,
      currentIndex: (state.currentIndex + 1) % state.values.length,
    ));
  }

  void previousOverlap() {
    if (state.values.isEmpty) return;

    final previousIndex =
        (state.values.length + state.currentIndex - 1) % state.values.length;

    emit(ListValuesScrollState(
      values: state.values,
      currentIndex: previousIndex,
    ));
  }
}

class ListValuesScrollState<T> {
  ListValuesScrollState({
    required this.values,
    required this.currentIndex,
  });

  final List<T> values;
  final int currentIndex;

  T? get current => values.isNotEmpty ? values[currentIndex] : null;

  bool get hasNext => currentIndex < values.length - 1;
  bool get hasPrevious => currentIndex > 0;
}

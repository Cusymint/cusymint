import 'package:flutter_bloc/flutter_bloc.dart';

class ListValuesScrollCubit<T> extends Cubit<ListValuesScrollState<T>> {
  ListValuesScrollCubit({
    required List<T> values,
    int startingIndex = 0,
  })  : assert(startingIndex >= 0 && startingIndex < values.length),
        super(
          ListValuesScrollState(values: values, currentIndex: startingIndex),
        );

  void next() {
    if (state.hasNext) {
      emit(ListValuesScrollState(
        values: state.values,
        currentIndex: state.currentIndex + 1,
      ));
    }
  }

  void previous() {
    if (state.hasPrevious) {
      emit(ListValuesScrollState(
        values: state.values,
        currentIndex: state.currentIndex - 1,
      ));
    }
  }

  void nextOverlap() {
    emit(ListValuesScrollState(
      values: state.values,
      currentIndex: (state.currentIndex + 1) % state.values.length,
    ));
  }

  void previousOverlap() {
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

  T get current => values[currentIndex];

  bool get hasNext => currentIndex < values.length - 1;
  bool get hasPrevious => currentIndex > 0;
}

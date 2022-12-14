import 'package:cusymint_app/features/home/blocs/list_values_scroll_cubit.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class InputHistoryScrollCubit extends Cubit<InputHistoryScrollState> {
  InputHistoryScrollCubit({
    required List<String> history,
    required String current,
  })  : _listValuesScrollCubit =
            ListValuesScrollCubit(values: [current, ...history]),
        super(InputHistoryScrollState(
          currentIndex: 0,
          history: [current, ...history],
        ));

  ListValuesScrollCubit<String> _listValuesScrollCubit;

  void next() {
    _listValuesScrollCubit.nextOverlap();
    emit(InputHistoryScrollState(
      history: _listValuesScrollCubit.state.values,
      currentIndex: _listValuesScrollCubit.state.currentIndex,
    ));
  }

  void previous() {
    _listValuesScrollCubit.previousOverlap();
    emit(InputHistoryScrollState(
      history: _listValuesScrollCubit.state.values,
      currentIndex: _listValuesScrollCubit.state.currentIndex,
    ));
  }

  void updateCurrentValue(String value) {
    final newHistory = [
      value,
      ..._listValuesScrollCubit.state.values.skip(1).toList(),
    ];
    _listValuesScrollCubit = ListValuesScrollCubit(
      values: newHistory,
      startingIndex: 0,
    );
    emit(InputHistoryScrollState(
      history: newHistory,
      currentIndex: 0,
    ));
  }
}

class InputHistoryScrollState {
  InputHistoryScrollState({
    required this.history,
    required this.currentIndex,
  });

  final List<String> history;
  int currentIndex = 0;

  String get current => history[currentIndex];
}

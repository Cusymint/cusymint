import 'package:bloc_test/bloc_test.dart';
import 'package:cusymint_app/features/home/blocs/input_history_cubit.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  tearDown(() {
    SharedPreferences.setMockInitialValues({});
  });

  blocTest(
    'Emits new inputs in history',
    build: () => InputHistoryCubit(),
    act: (InputHistoryCubit cubit) {
      cubit.addInput('1');
      cubit.addInput('abc');
    },
    expect: () => [
      _isAInputHistoryState(['1']),
      _isAInputHistoryState(['abc', '1']),
    ],
  );

  blocTest(
    'Emits empty history after clearing',
    build: () => InputHistoryCubit(),
    act: (InputHistoryCubit cubit) {
      cubit.addInput('1');
      cubit.addInput('2');
      cubit.clearHistory();
    },
    expect: () => [
      _isAInputHistoryState(['1']),
      _isAInputHistoryState(['2', '1']),
      _isAInputHistoryState([]),
    ],
  );

  blocTest(
    'Doesn\'t emit when last value is repeated',
    build: () => InputHistoryCubit(),
    act: (InputHistoryCubit cubit) {
      cubit.addInput('1');
      cubit.addInput('2');
      cubit.addInput('2');
    },
    expect: () => [
      _isAInputHistoryState(['1']),
      _isAInputHistoryState(['2', '1']),
    ],
  );

  blocTest(
    'Can add to history after clearing',
    build: () => InputHistoryCubit(),
    act: (InputHistoryCubit cubit) {
      cubit.addInput('1');
      cubit.clearHistory();
      cubit.addInput('2');
    },
    expect: () => [
      _isAInputHistoryState(['1']),
      _isAInputHistoryState([]),
      _isAInputHistoryState(['2'])
    ],
  );

  blocTest(
    'Doesn\'t emit anything when stored history is empty',
    build: () => InputHistoryCubit(),
    expect: () => [],
  );

  blocTest(
    'Emits stored history on load',
    setUp: () async {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setStringList(InputHistoryCubit.historyKey, ['1', 'abcdef']);
    },
    build: () => InputHistoryCubit(),
    expect: () => [
      _isAInputHistoryState(['1', 'abcdef'])
    ],
  );
}

TypeMatcher<InputHistoryState> _isAInputHistoryState(List<String> history) =>
    isA<InputHistoryState>()
        .having((s) => s.history, 'history', equals(history));

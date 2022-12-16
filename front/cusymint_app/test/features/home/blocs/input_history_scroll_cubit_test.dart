import 'package:bloc_test/bloc_test.dart';
import 'package:cusymint_app/features/home/blocs/input_history_scroll_cubit.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final history = ['a', 'b', 'c', 'd', 'e'];
  const textFieldValue = 'x';

  blocTest(
    'Emits InputHistoryScrollState'
    'with first history item when next is called',
    build: () => InputHistoryScrollCubit(
      history: history,
      current: textFieldValue,
    ),
    act: (cubit) => cubit.next(),
    expect: () => [
      _isAInputHistoryScrollStateWithCurrentItem('a'),
    ],
  );

  blocTest(
    'Emits InputHistoryScrollStates'
    'with subsequent history items when multiple next are called',
    build: () => InputHistoryScrollCubit(
      history: history,
      current: textFieldValue,
    ),
    act: (cubit) {
      // 8 is arbitrary, but it's more than the length of the history
      for (int i = 0; i < 8; ++i) {
        cubit.next();
      }
    },
    expect: () => [
      _isAInputHistoryScrollStateWithCurrentItem('a'),
      _isAInputHistoryScrollStateWithCurrentItem('b'),
      _isAInputHistoryScrollStateWithCurrentItem('c'),
      _isAInputHistoryScrollStateWithCurrentItem('d'),
      _isAInputHistoryScrollStateWithCurrentItem('e'),
    ],
  );

  blocTest(
    'Doesn\'t emit InputHistoryScrollState'
    'with last history item when previous is called on first item',
    build: () => InputHistoryScrollCubit(
      history: history,
      current: textFieldValue,
    ),
    act: (cubit) => cubit.previous(),
    expect: () => [],
  );

  blocTest(
    'Doesn\'t emit InputHistoryScrollState'
    'with last history item when previous is called multiple times'
    'on first item',
    build: () => InputHistoryScrollCubit(
      history: history,
      current: textFieldValue,
    ),
    act: (cubit) {
      // 8 is arbitrary, but it's more than the length of the history
      for (int i = 0; i < 8; ++i) {
        cubit.previous();
      }
    },
    expect: () => [],
  );

  blocTest(
    'Emits InputHistoryScrollStates'
    'with history item and then current when going back and forth',
    build: () => InputHistoryScrollCubit(
      history: history,
      current: textFieldValue,
    ),
    act: (cubit) {
      cubit.next();
      cubit.previous();
    },
    expect: () => [
      _isAInputHistoryScrollStateWithCurrentItem('a'),
      _isAInputHistoryScrollStateWithCurrentItem('x'),
    ],
  );

  blocTest(
    'Overrides first item when history item is changed',
    build: () => InputHistoryScrollCubit(
      history: ['abc'],
      current: 'x',
    ),
    act: (cubit) {
      // go to history
      cubit.next();
      // update (this should point us again to the first item)
      cubit.updateCurrentValue('def');
      // check that the second item is still 'abc'
      cubit.next();
      // check that the last item is in fact 'abc' and don't move
      cubit.next();
      // check that the first item is now 'def'
      cubit.previous();
      // check that the list contains only two elements
      // overlap and check that the last item is 'abc'
      cubit.previous();
    },
    expect: () => [
      _isAInputHistoryScrollStateWithCurrentItem('abc'),
      _isAInputHistoryScrollStateWithCurrentItem('def'),
      _isAInputHistoryScrollStateWithCurrentItem('abc'),
      _isAInputHistoryScrollStateWithCurrentItem('def'),
    ],
  );
}

TypeMatcher<InputHistoryScrollState> _isAInputHistoryScrollStateWithCurrentItem(
  String currentItem,
) {
  return isA<InputHistoryScrollState>().having(
    (state) => state.current,
    'currentItem',
    currentItem,
  );
}

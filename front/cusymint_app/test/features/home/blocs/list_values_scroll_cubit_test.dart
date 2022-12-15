import 'package:bloc_test/bloc_test.dart';
import 'package:cusymint_app/features/home/blocs/list_values_scroll_cubit.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final values = [1, 2, 3, 4, 5];

  blocTest(
    'Emits ListValuesScrollState'
    'with currentIndex 1 when next is called',
    build: () => ListValuesScrollCubit(values: values),
    act: (cubit) => cubit.next(),
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
    ],
  );

  blocTest(
    'Emits ListValuesScrollState'
    'with currentIndex 1 when nextOverlap is called',
    build: () => ListValuesScrollCubit(values: values),
    act: (cubit) => cubit.next(),
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
    ],
  );

  blocTest(
    'Emits ListValuesScrollStates'
    'with changed current index when multiple next are called',
    build: () => ListValuesScrollCubit(values: values),
    act: (cubit) {
      for (int i = 0; i < 3; i++) {
        cubit.next();
      }
    },
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
      _isAListValuesScrollStateWith(values: values, currentIndex: 2),
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
    ],
  );

  blocTest(
    'Emits ListValuesScrollStates'
    'with changed current index when multiple nextOverlap are called',
    build: () => ListValuesScrollCubit(values: values),
    act: (cubit) {
      for (int i = 0; i < 3; i++) {
        cubit.nextOverlap();
      }
    },
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
      _isAListValuesScrollStateWith(values: values, currentIndex: 2),
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
    ],
  );

  blocTest(
    'Emits ListValuesScrollState'
    'with currentIndex 3 when previous is called',
    build: () => ListValuesScrollCubit(values: values, startingIndex: 4),
    act: (cubit) => cubit.previous(),
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
    ],
  );

  blocTest(
    'Emits ListValuesScrollState'
    'with currentIndex 3 when previousOverlap is called',
    build: () => ListValuesScrollCubit(values: values, startingIndex: 4),
    act: (cubit) => cubit.previousOverlap(),
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
    ],
  );

  blocTest(
    'Emits ListValuesScrollStates'
    'with changed current index when multiple previous are called',
    build: () => ListValuesScrollCubit(values: values, startingIndex: 4),
    act: (cubit) {
      for (int i = 0; i < 3; i++) {
        cubit.previous();
      }
    },
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
      _isAListValuesScrollStateWith(values: values, currentIndex: 2),
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
    ],
  );

  blocTest(
    'Emits ListValuesScrollStates'
    'with changed current index when multiple previousOverlap are called',
    build: () => ListValuesScrollCubit(values: values, startingIndex: 4),
    act: (cubit) {
      for (int i = 0; i < 3; i++) {
        cubit.previousOverlap();
      }
    },
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
      _isAListValuesScrollStateWith(values: values, currentIndex: 2),
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
    ],
  );

  blocTest(
    'Doesn\'t emit anything when next is called and currentIndex is last',
    build: () => ListValuesScrollCubit(values: values),
    act: (cubit) {
      for (int i = 0; i < 5; i++) {
        cubit.next();
      }
    },
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
      _isAListValuesScrollStateWith(values: values, currentIndex: 2),
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
      _isAListValuesScrollStateWith(values: values, currentIndex: 4),
    ],
  );

  blocTest(
    'Overlaps anything when nextOverlap is called and currentIndex is last',
    build: () => ListValuesScrollCubit(values: values),
    act: (cubit) {
      for (int i = 0; i < 6; i++) {
        cubit.nextOverlap();
      }
    },
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
      _isAListValuesScrollStateWith(values: values, currentIndex: 2),
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
      _isAListValuesScrollStateWith(values: values, currentIndex: 4),
      _isAListValuesScrollStateWith(values: values, currentIndex: 0),
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
    ],
  );

  blocTest(
    'Doesn\'t emit anything when previous is called and currentIndex is first',
    build: () => ListValuesScrollCubit(values: values, startingIndex: 4),
    act: (cubit) {
      for (int i = 0; i < 5; i++) {
        cubit.previous();
      }
    },
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
      _isAListValuesScrollStateWith(values: values, currentIndex: 2),
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
      _isAListValuesScrollStateWith(values: values, currentIndex: 0),
    ],
  );

  blocTest(
    'Overlaps anything when previousOverlap is called'
    'and currentIndex is first',
    build: () => ListValuesScrollCubit(values: values, startingIndex: 4),
    act: (cubit) {
      for (int i = 0; i < 6; i++) {
        cubit.previousOverlap();
      }
    },
    expect: () => [
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
      _isAListValuesScrollStateWith(values: values, currentIndex: 2),
      _isAListValuesScrollStateWith(values: values, currentIndex: 1),
      _isAListValuesScrollStateWith(values: values, currentIndex: 0),
      _isAListValuesScrollStateWith(values: values, currentIndex: 4),
      _isAListValuesScrollStateWith(values: values, currentIndex: 3),
    ],
  );

  blocTest(
    'Doesn\'t emit anything when no action is called',
    build: () => ListValuesScrollCubit(values: values),
    expect: () => [],
    verify: (bloc) => expect(
      bloc.state,
      _isAListValuesScrollStateWith(
        values: values,
        currentIndex: 0,
      ),
    ),
  );

  blocTest(
    'Doesn\'t emit anything when next is called and currentIndex isn\'t first',
    build: () => ListValuesScrollCubit(values: values, startingIndex: 4),
    act: (cubit) => cubit.next(),
    expect: () => [],
    verify: (bloc) => expect(
      bloc.state,
      _isAListValuesScrollStateWith(
        values: values,
        currentIndex: 4,
      ),
    ),
  );

  test('hasNext returns false when pointing at last element', () {
    final state = ListValuesScrollState(values: values, currentIndex: 4);
    expect(state.hasNext, isFalse);
  });

  test('hasNext returns true when pointing at not last element', () {
    final state = ListValuesScrollState(values: values, currentIndex: 3);
    expect(state.hasNext, isTrue);
  });

  test('hasPrevious returns false when pointing at first element', () {
    final state = ListValuesScrollState(values: values, currentIndex: 0);
    expect(state.hasPrevious, isFalse);
  });

  test('hasPrevious returns true when pointing at not first element', () {
    final state = ListValuesScrollState(values: values, currentIndex: 1);
    expect(state.hasPrevious, isTrue);
  });

  test('current returns current value', () {
    final state1 = ListValuesScrollState(values: values, currentIndex: 1);
    expect(state1.current, equals(2));
    final state2 = ListValuesScrollState(values: values, currentIndex: 2);
    expect(state2.current, equals(3));
  });
}

TypeMatcher<ListValuesScrollState> _isAListValuesScrollStateWith(
        {required List<int> values, required int currentIndex}) =>
    isA<ListValuesScrollState>()
        .having((state) => state.values, 'values', equals(values))
        .having(
          (state) => state.currentIndex,
          'currentIndex',
          equals(currentIndex),
        );

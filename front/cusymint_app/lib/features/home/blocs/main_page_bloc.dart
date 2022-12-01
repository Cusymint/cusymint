import 'dart:async';

import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_client/cusymint_client.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:rxdart/rxdart.dart';

class MainPageBloc extends Bloc<MainPageEvent, MainPageState> {
  MainPageBloc({required this.clientFactory}) : super(const InitialState()) {
    on<SolveRequested>(
      _solveIntegral,
      transformer: restartable(),
    );
    on<InputChanged>(
      _onInputChanged,
      transformer: debounceRestartable(const Duration(milliseconds: 200)),
    );
  }

  final ClientFactory clientFactory;
  CusymintClient get _client => clientFactory.client;

  FutureOr<void> _solveIntegral(
      SolveRequested event, Emitter<MainPageState> emit) async {
    emit(SolvingState(userInput: event.input));

    final watch = Stopwatch()..start();

    final request = Request(event.input);

    try {
      final response = await _client.solveIntegral(request);

      watch.stop();
      if (response.hasErrors) {
        // TODO: nice error handling
        emit(SolveErrorState(userInput: event.input, errors: []));
        return;
      }

      final duration = watch.elapsed;

      emit(SolvedState(
        userInput: state.userInput,
        inputInTex: response.inputInTex!,
        inputInUtf: response.inputInUtf!,
        outputInTex: response.outputInTex!,
        outputInUtf: response.outputInUtf!,
        duration: duration,
      ));
    } catch (e) {
      emit(SolveErrorState(userInput: event.input, errors: []));
    }
  }

  FutureOr<void> _onInputChanged(
      InputChanged event, Emitter<MainPageState> emit) async {
    String? previousInputInTex;
    String? previousInputInUtf;

    if (state is InterpretedState) {
      previousInputInTex = (state as InterpretedState).inputInTex;
      previousInputInUtf = (state as InterpretedState).inputInUtf;
    }

    emit(InterpretingState(
      userInput: event.input,
      previousInputInTex: previousInputInTex,
      previousInputInUtf: previousInputInUtf,
    ));

    final request = Request(event.input);

    try {
      final response = await _client.interpretIntegral(request);

      if (response.hasErrors) {
        // TODO: nice error handling
        emit(InterpretErrorState(userInput: event.input, errors: []));
        return;
      }

      emit(InterpretedState(
        userInput: event.input,
        inputInTex: response.inputInTex!,
        inputInUtf: response.inputInUtf!,
      ));
    } catch (e) {
      // TODO: nice error handling
      emit(InterpretErrorState(userInput: event.input, errors: []));
    }
  }
}

abstract class MainPageEvent {
  const MainPageEvent();
}

class InputChanged extends MainPageEvent {
  const InputChanged(this.input);
  final String input;
}

class SolveRequested extends MainPageEvent {
  const SolveRequested(this.input);
  final String input;
}

class ClearRequested extends MainPageEvent {
  const ClearRequested();
}

class MainPageState {
  const MainPageState({required this.userInput});

  final String userInput;
}

class InitialState extends MainPageState {
  const InitialState() : super(userInput: '');
}

class InterpretingState extends MainPageState {
  const InterpretingState({
    required super.userInput,
    this.previousInputInTex,
    this.previousInputInUtf,
  });

  bool get hasPreviousInput =>
      previousInputInTex != null && previousInputInUtf != null;

  final String? previousInputInTex;
  final String? previousInputInUtf;
}

class InterpretedState extends MainPageState {
  const InterpretedState({
    required super.userInput,
    required this.inputInTex,
    required this.inputInUtf,
  });

  final String inputInTex;
  final String inputInUtf;
}

class InterpretErrorState extends MainPageState {
  const InterpretErrorState({
    required super.userInput,
    required this.errors,
  });

  final List<String> errors;
}

class SolvingState extends MainPageState {
  const SolvingState({
    required super.userInput,
  });
}

class SolvedState extends MainPageState {
  const SolvedState({
    required super.userInput,
    required this.inputInTex,
    required this.inputInUtf,
    required this.outputInTex,
    required this.outputInUtf,
    required this.duration,
  });

  final String inputInTex;
  final String inputInUtf;
  final String outputInTex;
  final String outputInUtf;
  final Duration duration;
}

class SolveErrorState extends MainPageState {
  const SolveErrorState({
    required super.userInput,
    this.inputInTex,
    this.inputInUtf,
    required this.errors,
  });

  final String? inputInTex;
  final String? inputInUtf;

  final List<String> errors;
}

EventTransformer<Event> debounceSequential<Event>(Duration duration) {
  return (events, mapper) => events.debounceTime(duration).asyncExpand(mapper);
}

EventTransformer<Event> debounceRestartable<Event>(Duration duration) {
  return (events, mapper) => events.debounceTime(duration).switchMap(mapper);
}

EventTransformer<Event> restartable<Event>() {
  return (events, mapper) => events.switchMap(mapper);
}

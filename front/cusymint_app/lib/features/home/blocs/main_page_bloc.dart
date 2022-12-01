import 'dart:async';

import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_client/cusymint_client.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:rxdart/rxdart.dart';

class MainPageBloc extends Bloc<MainPageEvent, MainPageState> {
  MainPageBloc({required this.clientFactory}) : super(const MainPageState()) {
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
    emit(state.copyWith(isLoading: true, errors: [], userInput: event.input));

    final watch = Stopwatch()..start();

    final request = Request(event.input);

    try {
      final response = await _client.solveIntegral(request);

      watch.stop();

      // TODO: do something with duration
      final duration = watch.elapsed;

      emit(state.copyWith(
        isLoading: false,
        errors: response.errors,
        inputInTex: Wrapped(response.inputInTex),
        inputInUtf: Wrapped(response.inputInUtf),
        outputInTex: Wrapped(response.outputInTex),
        outputInUtf: Wrapped(response.outputInUtf),
      ));
    } catch (e) {
      // TODO: Handle error
      //emit(SolveErrorState(userInput: event.input, errors: []));
    }
  }

  FutureOr<void> _onInputChanged(
      InputChanged event, Emitter<MainPageState> emit) async {
    if (event.input.isEmpty) {
      emit(state.copyWith(
        userInput: event.input,
        errors: [],
        isLoading: false,
      ));
    }

    emit(state.copyWith(
      userInput: event.input,
      isLoading: true,
      errors: [],
      outputInTex: const Wrapped(null),
      outputInUtf: const Wrapped(null),
    ));

    final request = Request(event.input);

    try {
      final response = await _client.interpretIntegral(request);

      emit(state.copyWith(
        isLoading: false,
        errors: response.errors,
        inputInTex: Wrapped(response.inputInTex),
        inputInUtf: Wrapped(response.inputInUtf),
        outputInTex: Wrapped(response.outputInTex),
        outputInUtf: Wrapped(response.outputInUtf),
      ));
    } catch (e) {
      // TODO: nice error handling
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
  const MainPageState({
    this.isLoading = false,
    this.inputInTex,
    this.inputInUtf,
    this.outputInTex,
    this.outputInUtf,
    this.previousInputInTex,
    this.previousInputInUtf,
    this.errors = const [],
    this.userInput = '',
  });

  final String userInput;

  final bool isLoading;
  final List<ResponseError> errors;

  final String? inputInTex;
  final String? inputInUtf;

  final String? outputInTex;
  final String? outputInUtf;

  final String? previousInputInTex;
  final String? previousInputInUtf;

  bool get hasErrors => errors.isNotEmpty;
  bool get hasInput => inputInTex != null && inputInUtf != null;
  bool get hasOutput => outputInTex != null && outputInUtf != null;

  MainPageState copyWith({
    String? userInput,
    bool? isLoading,
    List<ResponseError>? errors,
    Wrapped<String?>? inputInTex,
    Wrapped<String?>? inputInUtf,
    Wrapped<String?>? outputInTex,
    Wrapped<String?>? outputInUtf,
  }) {
    return MainPageState(
      isLoading: isLoading ?? this.isLoading,
      inputInTex: inputInTex != null ? inputInTex.value : this.inputInTex,
      inputInUtf: inputInUtf != null ? inputInUtf.value : this.inputInUtf,
      outputInTex: outputInTex != null ? outputInTex.value : this.outputInTex,
      outputInUtf: outputInUtf != null ? outputInUtf.value : this.outputInUtf,
      previousInputInTex:
          inputInTex != null ? this.inputInTex : previousInputInTex,
      previousInputInUtf:
          inputInUtf != null ? this.inputInUtf : previousInputInUtf,
      errors: errors ?? this.errors,
      userInput: userInput ?? this.userInput,
    );
  }
}

class Wrapped<T> {
  const Wrapped(this.value);
  final T value;
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

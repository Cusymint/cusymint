import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:shared_preferences/shared_preferences.dart';

class InputHistoryCubit extends Cubit<InputHistoryState> {
  InputHistoryCubit() : super(const InputHistoryState(history: [])) {
    _loadAndEmitHistory();
  }

  static const String _historyKey = 'history_key';

  void addInput(String input) async {
    final history = state.history;
    final newHistory = [...history, input];
    _updateStoredHistory(newHistory);
    emit(InputHistoryState(history: newHistory));
  }

  void clearHistory() {
    _updateStoredHistory([]);
    emit(const InputHistoryState(history: []));
  }

  void _updateStoredHistory(List<String> history) async {
    final prefs = await SharedPreferences.getInstance();
    prefs.setStringList(_historyKey, history);
  }

  void _loadAndEmitHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final history = prefs.getStringList(_historyKey);

    if (history != null && history.isNotEmpty) {
      emit(InputHistoryState(history: history));
    }
  }
}

class InputHistoryState {
  const InputHistoryState({
    required this.history,
  });

  /// List of inputs that were used in the past.
  /// The most recent input is at the end of the list.
  final List<String> history;
}

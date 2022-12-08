import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:shared_preferences/shared_preferences.dart';

class InputHistoryCubit extends Cubit<InputHistoryState> {
  InputHistoryCubit() : super(const InputHistoryState(history: [])) {
    _loadAndEmitHistory();
  }

  static const String _historyKey = 'history_key';

  void addInput(String input) async {
    final history = state.history;
    history.add(input);
    _updateStoredHistory(history);
    emit(InputHistoryState(history: history));
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
    final history = prefs.getStringList(_historyKey) ?? [];
    emit(InputHistoryState(history: history));
  }
}

class InputHistoryState {
  const InputHistoryState({
    required this.history,
  });

  final List<String> history;
}

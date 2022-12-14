import 'package:auto_route/auto_route.dart';
import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_app/features/home/blocs/input_history_cubit.dart';
import 'package:cusymint_app/features/home/blocs/input_history_scroll_cubit.dart';
import 'package:cusymint_app/features/home/blocs/main_page_bloc.dart';
import 'package:cusymint_app/features/home/utils/client_error_translator.dart';
import 'package:cusymint_app/features/navigation/navigation.dart';
import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:share_plus/share_plus.dart';

class HomePage extends StatelessWidget {
  const HomePage({
    super.key,
    this.isTextSelected = false,
    this.initialExpression,
  });

  final bool isTextSelected;
  final String? initialExpression;

  @override
  Widget build(BuildContext context) {
    final mainPageBloc = MainPageBloc(
      clientFactory: ClientFactory.of(context),
      initialExpression: initialExpression,
    );

    final historyCubit = InputHistoryCubit();

    return HomeBody(
      mainPageBloc: mainPageBloc,
      historyCubit: historyCubit,
      isTextSelected: isTextSelected,
    );
  }
}

class HomeBody extends StatefulWidget {
  const HomeBody({
    super.key,
    required this.mainPageBloc,
    required this.isTextSelected,
    required this.historyCubit,
  });

  final bool isTextSelected;
  final MainPageBloc mainPageBloc;
  final InputHistoryCubit historyCubit;

  @override
  State<HomeBody> createState() => _HomeBodyState();
}

class _HomeBodyState extends State<HomeBody> {
  final _fToast = FToast();
  final _controller = TextEditingController();

  @override
  void initState() {
    super.initState();
    _fToast.init(context);
    _controller.text = widget.mainPageBloc.initialExpression ?? '';
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return CuScaffold(
      drawer: WiredDrawer(context: context),
      appBar: CuAppBar(
        actions: [
          _HistoryIconButton(
            historyCubit: widget.historyCubit,
            mainPageBloc: widget.mainPageBloc,
            controller: _controller,
          ),
        ],
      ),
      body: Center(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          mainAxisSize: MainAxisSize.max,
          children: [
            _MainPageInput(
              controller: _controller,
              mainPageBloc: widget.mainPageBloc,
              historyCubit: widget.historyCubit,
              isTextSelected: widget.isTextSelected,
            ),
            BlocBuilder<MainPageBloc, MainPageState>(
              bloc: widget.mainPageBloc,
              builder: (context, state) {
                return CuAnimatedHomeCard(
                  inputInTex: state.inputInTex ?? state.previousInputInTex,
                  outputInTex: state.outputInTex,
                  isLoading: state.isLoading,
                  hasCriticalErrors: state.errors.isNotEmpty,
                  errors: ClientErrorTranslator.translateList(state.errors),
                  buttonRowCallbacks: state.hasOutput
                      ? CuButtonRowCallbacks(
                          onCopyTex: () async => _copyTexToClipboard(
                            state.inputInTex!,
                            state.outputInTex!,
                          ),
                          onCopyUtf: () async => _copyUtfToClipboard(
                            state.inputInUtf!,
                            state.outputInUtf!,
                          ),
                          onShareUtf: () async => _shareUtf(
                            state.inputInUtf!,
                            state.outputInUtf!,
                          ),
                        )
                      : null,
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _copyUtfToClipboard(
    String inputInUtf,
    String outputInUtf,
  ) async {
    _showCopyToClipboardToast();
    await Clipboard.setData(ClipboardData(text: '$inputInUtf = $outputInUtf'));
  }

  Future<void> _copyTexToClipboard(
    String inputInTex,
    String outputInTex,
  ) async {
    _showCopyToClipboardToast();
    await Clipboard.setData(ClipboardData(text: '$inputInTex = $outputInTex'));
  }

  Future<void> _shareUtf(
    String inputInUtf,
    String outputInUtf,
  ) async {
    await Share.share('$inputInUtf = $outputInUtf');
  }

  void _showCopyToClipboardToast() {
    _fToast.showToast(
      child: CuToast.success(message: Strings.copiedToClipboard.tr()),
      gravity: ToastGravity.BOTTOM,
      toastDuration: const Duration(seconds: 2),
    );
  }
}

class _HistoryIconButton extends StatelessWidget {
  const _HistoryIconButton({
    Key? key,
    required this.controller,
    required this.historyCubit,
    required this.mainPageBloc,
  }) : super(key: key);

  final InputHistoryCubit historyCubit;
  final MainPageBloc mainPageBloc;
  final TextEditingController controller;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      onPressed: () {
        showDialog(
          context: context,
          builder: (context) {
            return BlocBuilder<InputHistoryCubit, InputHistoryState>(
              bloc: historyCubit,
              builder: (context, state) => CuHistoryDialog(
                onClearHistoryPressed: () => historyCubit.clearHistory(),
                onCancelPressed: () => Navigator.of(context).pop(),
                historyItems: [
                  for (final item in state.history)
                    CuHistoryItem(
                      item,
                      onTap: () {
                        controller.text = item;
                        mainPageBloc.add(InputChanged(item));
                        Navigator.of(context).pop();
                      },
                    )
                ],
              ),
            );
          },
        );
      },
      icon: const Icon(Icons.history),
    );
  }
}

class _MainPageInput extends StatelessWidget {
  const _MainPageInput({
    Key? key,
    required this.mainPageBloc,
    required this.historyCubit,
    required this.controller,
    required this.isTextSelected,
  }) : super(key: key);

  final MainPageBloc mainPageBloc;
  final InputHistoryCubit historyCubit;
  final TextEditingController controller;
  final bool isTextSelected;

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<InputHistoryCubit, InputHistoryState>(
      bloc: historyCubit,
      builder: (context, state) {
        final listCubit = InputHistoryScrollCubit(
          history: historyCubit.state.history,
          current: controller.text,
        );

        return Padding(
          padding: const EdgeInsets.all(8.0),
          child: SizedBox(
            width: 400,
            child: Hero(
              tag: 'input',
              child: Column(
                children: [
                  CuText.med14(Strings.enterIntegral.tr()),
                  _ActionsDetector(
                    onDownPressed: () {
                      listCubit.previous();
                    },
                    onUpPressed: () {
                      listCubit.next();
                    },
                    onEscapePressed: () {
                      _clearOrPop(context, listCubit);
                    },
                    child: BlocListener<InputHistoryScrollCubit,
                        InputHistoryScrollState>(
                      bloc: listCubit,
                      listenWhen: (previous, current) =>
                          previous.currentIndex != current.currentIndex,
                      listener: (context, state) {
                        controller.text = state.current;
                        controller.selection = TextSelection.fromPosition(
                          TextPosition(offset: state.current.length),
                        );
                      },
                      child: CuTextField(
                        autofocus: isTextSelected,
                        onSubmitted: (_) => _submit(),
                        onChanged: (newText) {
                          listCubit.updateCurrentValue(newText);
                          mainPageBloc.add(InputChanged(newText));
                        },
                        prefixIcon: IconButton(
                          onPressed: () => _clearOrPop(context, listCubit),
                          icon: Icon(
                            Icons.clear,
                            color: CuColors.of(context).mintDark,
                          ),
                        ),
                        suffixIcon: IconButton(
                          onPressed: _submit,
                          icon: Icon(
                            Icons.send,
                            color: CuColors.of(context).mintDark,
                          ),
                        ),
                        controller: controller,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  void _submit() {
    if (controller.text.isNotEmpty) {
      final input = controller.text;

      historyCubit.addInput(input);
      mainPageBloc.add(SolveRequested(input));
    }
  }

  void _clearOrPop(BuildContext context, InputHistoryScrollCubit listCubit) {
    if (controller.text.isNotEmpty) {
      controller.clear();
      mainPageBloc.add(const ClearRequested());
      listCubit.updateCurrentValue('');
      return;
    }

    context.router.popUntilRoot();
  }
}

class _ActionsDetector extends StatelessWidget {
  const _ActionsDetector({
    required this.child,
    required this.onUpPressed,
    required this.onDownPressed,
    required this.onEscapePressed,
  });

  final Widget child;
  final VoidCallback onUpPressed;
  final VoidCallback onDownPressed;
  final VoidCallback onEscapePressed;

  @override
  Widget build(BuildContext context) {
    return Shortcuts(
      shortcuts: <ShortcutActivator, Intent>{
        LogicalKeySet(LogicalKeyboardKey.arrowUp): const _UpIntent(),
        LogicalKeySet(LogicalKeyboardKey.arrowDown): const _DownIntent(),
        LogicalKeySet(LogicalKeyboardKey.escape): const _EscapeIntent(),
      },
      child: Actions(
        actions: <Type, Action<Intent>>{
          _UpIntent: CallbackAction<_UpIntent>(
            onInvoke: (intent) => onUpPressed(),
          ),
          _DownIntent: CallbackAction<_DownIntent>(
            onInvoke: (intent) => onDownPressed(),
          ),
          _EscapeIntent: CallbackAction<_EscapeIntent>(
            onInvoke: (intent) => onEscapePressed(),
          ),
        },
        child: child,
      ),
    );
  }
}

class _EscapeIntent extends Intent {
  const _EscapeIntent();
}

class _UpIntent extends Intent {
  const _UpIntent();
}

class _DownIntent extends Intent {
  const _DownIntent();
}

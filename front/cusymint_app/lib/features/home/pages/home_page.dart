import 'package:cusymint_app/features/home/blocs/client_cubit.dart';
import 'package:cusymint_app/features/navigation/navigation.dart';
import 'package:cusymint_app/features/tex_rendering/widgets/tex_view.dart';
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
  });

  final bool isTextSelected;

  @override
  Widget build(BuildContext context) {
    final clientCubit = BlocProvider.of<ClientCubit>(context);

    return HomeBody(
      clientCubit: clientCubit,
      isTextSelected: isTextSelected,
    );
  }
}

class HomeBody extends StatefulWidget {
  const HomeBody({
    super.key,
    required this.clientCubit,
    required this.isTextSelected,
  });

  final bool isTextSelected;
  final ClientCubit clientCubit;

  @override
  State<HomeBody> createState() => _HomeBodyState();
}

class _HomeBodyState extends State<HomeBody> {
  final _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<ClientCubit, ClientState>(
      bloc: widget.clientCubit,
      builder: (context, state) {
        return CuScaffold(
          drawer: WiredDrawer(context: context),
          appBar: CuAppBar(),
          body: Center(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              mainAxisSize: MainAxisSize.max,
              children: [
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: SizedBox(
                    width: 400,
                    child: Hero(
                      tag: 'input',
                      child: Column(
                        children: [
                          CuText.med14(Strings.enterIntegral.tr()),
                          CuTextField(
                            autofocus: widget.isTextSelected,
                            onSubmitted: (submittedText) {
                              widget.clientCubit.solveIntegral(submittedText);
                            },
                            prefixIcon: IconButton(
                              onPressed: () {
                                _controller.clear();
                                widget.clientCubit.reset();
                              },
                              icon: Icon(
                                Icons.clear,
                                color: CuColors.of(context).mintDark,
                              ),
                            ),
                            suffixIcon: IconButton(
                              onPressed: () {
                                if (_controller.text.isNotEmpty) {
                                  widget.clientCubit
                                      .solveIntegral(_controller.text);
                                }
                              },
                              icon: Icon(
                                Icons.send,
                                color: CuColors.of(context).mintDark,
                              ),
                            ),
                            controller: _controller,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                if (state is ClientLoading) const CuInterpretingView(),
                if (state is ClientSuccess)
                  _SuccessBody(
                    inputInTex: state.inputInTex,
                    inputInUtf: state.inputInUtf,
                    outputInTex: state.outputInTex,
                    outputInUtf: state.outputInUtf,
                    duration: state.duration,
                  ),
                if (state is ClientFailure)
                  _FailureBody(
                    errors: state.errors,
                  ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class _SuccessBody extends StatefulWidget {
  const _SuccessBody({
    required this.inputInUtf,
    required this.inputInTex,
    required this.outputInUtf,
    required this.outputInTex,
    required this.duration,
  });

  final String inputInUtf;
  final String inputInTex;
  final String outputInUtf;
  final String outputInTex;
  final Duration duration;

  @override
  State<_SuccessBody> createState() => _SuccessBodyState();
}

class _SuccessBodyState extends State<_SuccessBody> {
  final _scrollController = ScrollController();
  final _fToast = FToast();

  @override
  void initState() {
    super.initState();
    _fToast.init(context);
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          Strings.foundResult.tr(
            namedArgs: {
              'timeInMs': widget.duration.inMilliseconds.toString(),
            },
          ),
        ),
        Center(
          child: CuCard(
            child: Column(
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8.0),
                  child: Scrollbar(
                    thumbVisibility: true,
                    controller: _scrollController,
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      controller: _scrollController,
                      child: Padding(
                        padding: const EdgeInsets.all(16),
                        child: TexView(
                          '${widget.inputInTex} = ${widget.outputInTex}',
                          fontScale: 2,
                        ),
                      ),
                    ),
                  ),
                ),
                ButtonBar(
                  alignment: MainAxisAlignment.end,
                  children: [
                    IconButton(
                      onPressed: () async => await _shareUtf(),
                      // TODO: replace with cusymint icons
                      icon: const Icon(Icons.share),
                    ),
                    IconButton(
                      onPressed: () async => await _copyTexToClipboard(),
                      // TODO: replace with cusymint icons
                      icon: const Icon(Icons.copy),
                    ),
                    IconButton(
                      onPressed: () async => await _copyUtfToClipboard(),
                      // TODO: replace with cusymint icons
                      icon: const Icon(Icons.copy_sharp),
                    ),
                  ],
                )
              ],
            ),
          ),
        ),
      ],
    );
  }

  Future<void> _copyUtfToClipboard() async {
    _showCopyToClipboardToast();
    await Clipboard.setData(
      ClipboardData(text: '${widget.inputInUtf} = ${widget.outputInUtf}'),
    );
  }

  Future<void> _copyTexToClipboard() async {
    _showCopyToClipboardToast();
    await Clipboard.setData(
      ClipboardData(text: '${widget.inputInTex} = ${widget.outputInTex}'),
    );
  }

  Future<void> _shareUtf() async {
    await Share.share('${widget.inputInUtf} = ${widget.outputInUtf}');
  }

  void _showCopyToClipboardToast() {
    _fToast.showToast(
      child: CuToast.success(message: Strings.copiedToClipboard.tr()),
      gravity: ToastGravity.BOTTOM,
      toastDuration: const Duration(seconds: 2),
    );
  }
}

class _FailureBody extends StatelessWidget {
  const _FailureBody({
    required this.errors,
  });

  final List<String> errors;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        for (final error in errors) CuText.med14(error),
        CuElevatedButton(
          onPressed: () => context.read<ClientCubit>().reset(),
          text: Strings.retry.tr(),
        ),
      ],
    );
  }
}

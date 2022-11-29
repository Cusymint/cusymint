import 'package:auto_route/auto_route.dart';
import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_app/features/home/blocs/main_page_bloc.dart';
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
    final mainPageBloc = MainPageBloc(clientFactory: ClientFactory.of(context));

    return HomeBody(
      mainPageBloc: mainPageBloc,
      isTextSelected: isTextSelected,
    );
  }
}

class HomeBody extends StatefulWidget {
  const HomeBody({
    super.key,
    required this.mainPageBloc,
    required this.isTextSelected,
  });

  final bool isTextSelected;
  final MainPageBloc mainPageBloc;

  @override
  State<HomeBody> createState() => _HomeBodyState();
}

class _HomeBodyState extends State<HomeBody> {
  final _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
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
                          if (submittedText.isNotEmpty) {
                            widget.mainPageBloc.add(
                              SolveRequested(submittedText),
                            );
                          }
                        },
                        onChanged: (newText) {
                          widget.mainPageBloc.add(
                            InputChanged(newText),
                          );
                        },
                        prefixIcon: IconButton(
                          onPressed: () {
                            if (_controller.text.isNotEmpty) {
                              _controller.clear();
                              return;
                            }

                            context.router.popUntilRoot();
                          },
                          icon: Icon(
                            Icons.clear,
                            color: CuColors.of(context).mintDark,
                          ),
                        ),
                        suffixIcon: IconButton(
                          onPressed: () {
                            if (_controller.text.isNotEmpty) {
                              widget.mainPageBloc.add(
                                SolveRequested(_controller.text),
                              );
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
            BlocBuilder<MainPageBloc, MainPageState>(
              bloc: widget.mainPageBloc,
              builder: (context, state) {
                if (state is InterpretingState) {
                  return const CuInterpretLoadingCard();
                }

                if (state is InterpretedState) {
                  return CuInterpretResultCard(
                    child: TexView(state.inputInTex),
                  );
                }

                if (state is SolvingState) {
                  // TODO: implement
                  return const CuInterpretLoadingCard();
                }

                if (state is SolvedState) {
                  return CuSolveResultCard(
                    copyTex: () {},
                    copyUtf: () {},
                    shareUtf: () {},
                    solvingDuration: state.duration,
                    child: TexView(
                      '${state.inputInTex} = ${state.outputInTex}',
                    ),
                  );
                }

                return Container();
              },
            ),
          ],
        ),
      ),
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

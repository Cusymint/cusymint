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
  final _fToast = FToast();

  @override
  void initState() {
    super.initState();
    _fToast.init(context);
  }

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
            _MainPageInput(
              mainPageBloc: widget.mainPageBloc,
              isTextSelected: widget.isTextSelected,
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
                    copyTex: () async => _copyTexToClipboard(state),
                    copyUtf: () async => _copyUtfToClipboard(state),
                    shareUtf: () async => _shareUtf(state),
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

  Future<void> _copyUtfToClipboard(SolvedState state) async {
    _showCopyToClipboardToast();
    await Clipboard.setData(
      ClipboardData(text: '${state.inputInUtf} = ${state.outputInUtf}'),
    );
  }

  Future<void> _copyTexToClipboard(SolvedState state) async {
    _showCopyToClipboardToast();
    await Clipboard.setData(
      ClipboardData(text: '${state.inputInTex} = ${state.outputInTex}'),
    );
  }

  Future<void> _shareUtf(SolvedState state) async {
    await Share.share('${state.inputInUtf} = ${state.outputInUtf}');
  }

  void _showCopyToClipboardToast() {
    _fToast.showToast(
      child: CuToast.success(message: Strings.copiedToClipboard.tr()),
      gravity: ToastGravity.BOTTOM,
      toastDuration: const Duration(seconds: 2),
    );
  }
}

class _MainPageInput extends StatefulWidget {
  const _MainPageInput({
    Key? key,
    required this.mainPageBloc,
    required this.isTextSelected,
  }) : super(key: key);

  final MainPageBloc mainPageBloc;
  final bool isTextSelected;

  @override
  State<_MainPageInput> createState() => _MainPageInputState();
}

class _MainPageInputState extends State<_MainPageInput> {
  final _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
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
    );
  }
}

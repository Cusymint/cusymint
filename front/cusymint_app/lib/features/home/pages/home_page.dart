import 'package:cusymint_app/features/home/blocs/client_cubit.dart';
import 'package:cusymint_app/features/tex_rendering/widgets/tex_view.dart';
import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _controller = TextEditingController();

  @override
  Widget build(BuildContext context) {
    final client = CusymintClientMock(
      fakeResponse: ResponseMockFactory.defaultResponse,
    );

    final clientCubit = ClientCubit(client: client);

    return BlocBuilder<ClientCubit, ClientState>(
      bloc: clientCubit,
      builder: (context, state) {
        return SafeArea(
          child: Scaffold(
            floatingActionButton: FloatingActionButton(
              onPressed: () {
                clientCubit.reset();
              },
              child: const Icon(Icons.refresh),
            ),
            body: Center(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                mainAxisSize: MainAxisSize.max,
                children: [
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: SizedBox(
                      width: 400,
                      child: CuTextField(
                        controller: _controller,
                      ),
                    ),
                  ),
                  if (state is ClientInitial)
                    ElevatedButton(
                      onPressed: () async {
                        final integralToBeSolved = _controller.text;
                        await clientCubit.solveIntegral(integralToBeSolved);
                      },
                      child: const Text('Solve integral'),
                    ),
                  if (state is ClientLoading) const _LoadingBody(),
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
  final ScrollController _scrollController = ScrollController();

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
            child: Scrollbar(
              thumbVisibility: true,
              controller: _scrollController,
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                controller: _scrollController,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: TexView('${widget.inputInTex} = ${widget.outputInTex}',
                      fontScale: 2),
                ),
              ),
            ),
          ),
        ),
      ],
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
        for (final error in errors) Text(error),
        ElevatedButton(
          onPressed: () => context.read<ClientCubit>().reset(),
          child: const Text('Reset'),
        ),
      ],
    );
  }
}

class _LoadingBody extends StatelessWidget {
  const _LoadingBody();

  @override
  Widget build(BuildContext context) {
    return const CircularProgressIndicator();
  }
}

import 'package:cusymint_app/features/home/blocs/client_cubit.dart';
import 'package:cusymint_app/features/tex_rendering/widgets/tex_view.dart';
import 'package:cusymint_client_mock/cusymint_client_mock.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    final client = CusymintClientMock(
      fakeResponse: ResponseMockFactory.defaultResponse,
    );

    final clientCubit = ClientCubit(client: client);

    return BlocBuilder<ClientCubit, ClientState>(
      bloc: clientCubit,
      builder: (context, state) {
        return Scaffold(
          floatingActionButton: FloatingActionButton(
            onPressed: () {
              clientCubit.reset();
            },
            child: const Icon(Icons.refresh),
          ),
          body: Column(
            children: [
              const Padding(
                padding: EdgeInsets.all(8.0),
                child: CuTextField(),
              ),
              if (state is ClientInitial)
                ElevatedButton(
                  onPressed: () =>
                      clientCubit.solveIntegral('x^2 + sin(x) / cos(x)'),
                  child: const Text('Solve integral'),
                ),
              if (state is ClientLoading) const _LoadingBody(),
              if (state is ClientSuccess)
                _SuccessBody(
                  response: state.response,
                ),
              if (state is ClientFailure)
                _FailureBody(
                  errors: state.errors,
                ),
            ],
          ),
        );
      },
    );
  }
}

class _SuccessBody extends StatelessWidget {
  const _SuccessBody({super.key, required this.response});

  final Response response;

  @override
  Widget build(BuildContext context) {
    return CuCard(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: TexView(response.outputInTex!, fontScale: 2),
      ),
    );
  }
}

class _FailureBody extends StatelessWidget {
  const _FailureBody({
    super.key,
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
  const _LoadingBody({super.key});

  @override
  Widget build(BuildContext context) {
    return const CircularProgressIndicator();
  }
}

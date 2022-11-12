import 'package:cusymint_app/features/client/client_factory.dart';
import 'package:cusymint_app/features/navigation/navigation.dart';
import 'package:cusymint_app/features/settings/blocs/client_url_cubit.dart';
import 'package:cusymint_l10n/cusymint_l10n.dart';
import 'package:cusymint_ui/cusymint_ui.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final clientFactory = ClientFactory.of(context);
    final clientUrlCubit = ClientUrlCubit(clientFactory: clientFactory);

    return _SettingsPageBody(
      clientUrlCubit: clientUrlCubit,
      clientFactory: clientFactory,
    );
  }
}

class _SettingsPageBody extends StatelessWidget {
  const _SettingsPageBody({
    Key? key,
    required this.clientUrlCubit,
    required this.clientFactory,
  }) : super(key: key);

  final ClientUrlCubit clientUrlCubit;
  final ClientFactory clientFactory;

  @override
  Widget build(BuildContext context) {
    return SettingsPageTemplate<Locale>(
      drawer: WiredDrawer(context: context),
      chosenLanguage: Strings.languageCurrent.tr(),
      // TODO: fix
      ipAddress: clientFactory.uri.toString(),
      onIpAddressTap: () {
        showDialog(
          context: context,
          builder: (context) => _UrlAlertDialog(clientUrlCubit: clientUrlCubit),
        );
      },
      onLicensesTap: () {
        showLicensePage(
          context: context,
          applicationIcon: CuLogo(
            color: CuColors.of(context).black,
          ),
          applicationName: '',
        );
      },
      selectedLocale: context.locale,
      languageMenuItems: [
        _LocalePopupMenuItem(
          context: context,
          locale: const Locale('en'),
          language: Strings.languageEnglish.tr(),
        ),
        _LocalePopupMenuItem(
          context: context,
          locale: const Locale('pl'),
          language: Strings.languagePolish.tr(),
        ),
      ],
    );
  }
}

class _UrlAlertDialog extends StatelessWidget {
  const _UrlAlertDialog({
    required this.clientUrlCubit,
  });

  final ClientUrlCubit clientUrlCubit;

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<ClientUrlCubit, ClientUrlState>(
      bloc: clientUrlCubit,
      builder: (context, state) {
        return CuTextFieldAlertDialog(
          title: Strings.ipAddress.tr(),
          textField: CuTextField(
            keyboardType: TextInputType.url,
            onChanged: (newText) => clientUrlCubit.onChangedUrl(newText),
            autofocus: true,
          ),
          onOkPressed: state.isValid
              ? () {
                  clientUrlCubit.setUrl();
                  Navigator.of(context).pop();
                }
              : null,
          onCancelPressed: () {
            Navigator.of(context).pop();
          },
        );
      },
    );
  }
}

class _LocalePopupMenuItem extends PopupMenuItem<Locale> {
  _LocalePopupMenuItem({
    required this.locale,
    required this.context,
    required this.language,
  }) : super(
          value: locale,
          child: CuText.med14(language),
          onTap: () async {
            await context.setLocale(locale);
          },
        );

  final Locale locale;
  final String language;
  final BuildContext context;
}

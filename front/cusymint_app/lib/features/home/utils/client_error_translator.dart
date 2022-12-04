import 'package:cusymint_client/cusymint_client.dart';
import 'package:cusymint_l10n/cusymint_l10n.dart';

class ClientErrorTranslator {
  static String translate(ResponseError error) {
    if (error is UnexpectedEndOfInputError) {
      return Strings.errorUnexpectedEndOfInput.tr();
    }

    if (error is UnexpectedTokenError) {
      return Strings.errorUnexpectedToken.tr(args: [error.token]);
    }

    if (error is NoSolutionFoundError) {
      return Strings.errorNoSolutionFound.tr();
    }

    if (error is InternalError) {
      return Strings.errorInternal.tr();
    }

    return Strings.errorUnknown.tr(args: [error.errorMessage]);
  }

  static List<String> translateList(List<ResponseError> errors) {
    return errors.map((ResponseError e) => translate(e)).toList();
  }
}

FROM fischerscode/flutter-sudo:latest AS build
USER flutter
WORKDIR /app

RUN flutter config --enable-web

# Requires to have context set to the root of the project
COPY ../../front /app/front
COPY ../../melos.yaml /app/melos.yaml

RUN sudo chown -R flutter:flutter /app
# RUN sudo chown -R flutter:flutter /opt/flutter

# Install melos
RUN flutter pub global activate melos
ENV PATH="$PATH:/home/flutter/.pub-cache/bin"

RUN melos --version
RUN git --version

# Build the web app
RUN melos bootstrap
RUN melos build:web

# Build the image
FROM nginx:stable-alpine
COPY --from=build /app/front/cusymint_app/build/web /usr/share/nginx/html

FROM alpine:3.19

WORKDIR /ios
COPY . /ios


CMD ["ls", "-l", "/ios"]
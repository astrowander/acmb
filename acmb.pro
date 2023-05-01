TEMPLATE = subdirs

SUBDIRS += \
    ./acmb-lib.pro \
    App/acmb-app.pro \
    Client/acmb-client.pro \
    Cuda/acmb-cuda.pro \
    Server/acmb-server.pro \
    Tests/acmb-tests.pro

Tests/acmb-tests.pro.depedns = acmb-lib.pro
App/acmb-app.pro.depedns = acmb-lib.pro

TEMPLATE = subdirs

SUBDIRS += \
    ./acmb-lib.pro \
    App/acmb-app.pro \
    Cuda/acmb-cuda.pro \
    GUI/acmb-gui.pro \
    Tests/acmb-tests.pro

Tests/acmb-tests.pro.depedns = acmb-lib.pro
App/acmb-app.pro.depedns = acmb-lib.pro

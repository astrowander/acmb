TEMPLATE = subdirs

SUBDIRS += \
    ./acmb-lib.pro \
    App/acmb-app.pro \
    Tests/acmb-tests.pro

Tests/acmb-tests.pro.depedns = acmb-lib.pro

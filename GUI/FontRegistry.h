#pragma once
#include "./../Core/macros.h"

struct ImFont;

ACMB_GUI_NAMESPACE_BEGIN

struct FontRegistry
{
    ImFont* byDefault;
    ImFont* icons;
    ImFont* bold;
    ImFont* big;
    ImFont* bigBold;

private:
    FontRegistry( float scaling );

    FontRegistry( const FontRegistry& ) = delete;
    FontRegistry( FontRegistry&& ) = delete;

public:
    static FontRegistry& Instance();
};

ACMB_GUI_NAMESPACE_END

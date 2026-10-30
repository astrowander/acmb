#pragma once

#include "../Core/macros.h"
#include <stdexcept>
#include <string>

ACMB_GUI_NAMESPACE_BEGIN

// Keep the tool action testable without loading decoders or the application singleton.
template<class Interface, class Input, class ReadDark, class Calculate>
void AdjustDarkFrameMultiplier(Interface& ui, Input input, int darkFrameCount,
    ReadDark readDark, Calculate calculate, float& multiplier,
    std::string& error, bool& showError)
{
    ui.LockInterface();
    try
    {
        if (!input || input->GetTaskCount() == 0)
            throw std::runtime_error("No input frames for adjusting the dark frame multiplier");
        if (darkFrameCount == 0)
            throw std::runtime_error("No dark frames for adjusting the dark frame multiplier");

        const auto lightFrame = input->RunTaskAndReportProgress(0);
        if (!lightFrame)
            throw std::runtime_error(lightFrame.error());
        const auto darkFrame = readDark();
        if (!darkFrame)
            throw std::runtime_error(darkFrame.error());

        multiplier = calculate(lightFrame.value(), darkFrame.value());
    }
    catch (const std::exception& e)
    {
        error = e.what();
        showError = true;
    }
    ui.UnlockInterface();
}

ACMB_GUI_NAMESPACE_END

#pragma once
#include "ImGuiHelpers.h"
#include "PipelineElementWindow.h"
#include "Serializer.h"

ACMB_GUI_NAMESPACE_BEGIN

template<typename TransformType>
class SettingsInterpolationUser
{
    using Settings = typename TransformType::Settings;
    std::map<int, Settings> _mSettings;
    PipelineElementWindow* _pHostWindow = nullptr;

public:

    SettingsInterpolationUser(PipelineElementWindow* pHostWindow, Settings defaultSettings)
    : _pHostWindow( pHostWindow ) 
    {
        _mSettings[0] = defaultSettings;
    }

    void AddSettings(int index, const Settings& settings)
    {
        _mSettings[index] = settings;
    }

    Settings GetInterpolatedSettings(int frameIndex) const
    {
        if (_mSettings.empty())
            return Settings{};

        auto it = _mSettings.upper_bound(frameIndex);
        if ( it == _mSettings.end() )
        {
            return std::prev(it)->second;
        }

        auto nextIt = it--;
        if ( it == _mSettings.end() )
            return nextIt->second;

        return TransformType::Interpolate( it->second, nextIt->second, double(frameIndex - it->first) / double(nextIt->first - it->first) );
    }

    void DrawFrameCounter()
    {
        ImGui::Separator();

        int currentFrame = _pHostWindow->GetPreviewedFrameNumber();
        if ( UI::InputInt( "Frame #", &currentFrame, 1, 10, 0, std::max(0, int( _pHostWindow->GetTaskCount(/*update=*/true)) - 1), "Frame number", nullptr) )
        {
            _pHostWindow->OnPreviewedFrameNumberChanged(currentFrame);
        }

        auto it = _mSettings.find(currentFrame);
        if ( it != _mSettings.end() )
        {
            UI::Button("Commit KF", { 64, 0 }, [&]
            {
                _pHostWindow->OnKeyframeCommited();
            }, "Commit this keyframe", nullptr);
            
            // Can't delete the first keyframe
            if ( currentFrame != 0 )
            {
                ImGui::SameLine();

                UI::Button("Delete Keyframe", { 64, 0 }, [&]
                {
                    _mSettings.erase(it);
                }, "Delete this keyframe", nullptr);
            }
        }
        else
        {
            UI::Button("Add Keyframe", { -1, 0 }, [&]
            {
                _pHostWindow->OnKeyframeCommited();
            }, "Add this keyframe", nullptr );
        }

        UI::Button("Prev KF", { 64, 0 },
                   [&]
        {
            auto it = _mSettings.lower_bound(currentFrame);
            if ( it != _mSettings.begin() )
                --it;

            currentFrame = it->first;
            _pHostWindow->OnPreviewedFrameNumberChanged(currentFrame);
        }, "Previous keyframe", nullptr);

        ImGui::SameLine();

        UI::Button("Next KF", { 64, 0 },
                   [&]
        {
            auto it = _mSettings.upper_bound(currentFrame);
            if ( it != _mSettings.end() )
            {
                currentFrame = it->first;
                _pHostWindow->OnPreviewedFrameNumberChanged(currentFrame);
            }
        }, "Next keyframe", nullptr);
                
    }

    void Serialize(std::ostream& out) const
    {
        gui::Serialize(_mSettings, out);
    }

    void Deserialize(std::istream& in, int& remainingBytes)
    {
        _mSettings = gui::Deserialize<std::map<int, Settings>>(in, remainingBytes);
    }

    int GetSerializedStringSize() const
    {
        return
            gui::GetSerializedStringSize(_mSettings);
    }
};

ACMB_GUI_NAMESPACE_END

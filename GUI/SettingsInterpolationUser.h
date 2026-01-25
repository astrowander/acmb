#pragma once
#include "ImGuiHelpers.h"
#include "PipelineElementWindow.h"
#include "Serializer.h"

ACMB_GUI_NAMESPACE_BEGIN

class ISettingsInterpolationUser
{
public:
    virtual ~ISettingsInterpolationUser() = default;
    virtual void DrawFrameCounter() = 0;
};

template<typename TransformType>
class SettingsInterpolationUser : public ISettingsInterpolationUser
{
public:
    using Transform = TransformType;
    using Settings = typename TransformType::Settings;

private:
    std::map<int, Settings> _mSettings;
    PipelineElementWindow* _pHostWindow = nullptr;

protected:
    SettingsInterpolationUser(PipelineElementWindow* pHostWindow, Settings defaultSettings)
    : _pHostWindow( pHostWindow ) 
    {
        _mSettings[0] = defaultSettings;
    }

    void InsertOrAssignSettings(int index, const Settings& settings)
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

        const float buttonWidth = 150.0f;
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float totalWidth = buttonWidth * 2 + spacing;

        int currentFrame = _pHostWindow->GetPreviewedFrameNumber();
        auto it = _mSettings.find(currentFrame);
        if ( it != _mSettings.end() )
        {
            UI::Button("Commit Keyframe", { buttonWidth, 0 }, [&]
            {
                _pHostWindow->OnKeyframeCommited();
            }, "Commit this keyframe", nullptr);
            
            // Can't delete the first keyframe
            if ( currentFrame != 0 )
            {
                ImGui::SameLine();

                UI::Button("Delete Keyframe", { buttonWidth, 0 }, [&]
                {
                    _mSettings.erase(it);
                }, "Delete this keyframe", nullptr);
            }
        }
        else
        {
            UI::Button("Add Keyframe", { totalWidth, 0 }, [&]
            {
                _pHostWindow->OnKeyframeCommited();
            }, "Add this keyframe", nullptr );
        }

        UI::Button("Previous Keyframe", { buttonWidth, 0 },
                   [&]
        {
            auto it = _mSettings.lower_bound(currentFrame);
            if ( it != _mSettings.begin() )
                --it;

            currentFrame = it->first;
            _pHostWindow->OnPreviewedFrameNumberChanged(currentFrame);
        }, "Go to previous keyframe", nullptr);

        ImGui::SameLine();

        UI::Button("Next Keyframe", { buttonWidth, 0 },
                   [&]
        {
            auto it = _mSettings.upper_bound(currentFrame);
            if ( it != _mSettings.end() )
            {
                currentFrame = it->first;
                _pHostWindow->OnPreviewedFrameNumberChanged(currentFrame);
            }
        }, "Go to next keyframe", nullptr);
                
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

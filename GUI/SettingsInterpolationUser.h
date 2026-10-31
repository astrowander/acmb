#pragma once
#include "ImGuiHelpers.h"
#include "PipelineElementWindow.h"
#include "Serializer.h"
#include <vector>

ACMB_GUI_NAMESPACE_BEGIN

class ISettingsInterpolationUser
{
public:
    virtual ~ISettingsInterpolationUser() = default;
    virtual void DrawFrameCounter(float availableWidth) = 0;
    virtual void CutExtraKeyframes(int maxFrame) = 0;
    virtual std::vector<int> GetKeyframeNumbers() const = 0;
};

template<typename TransformType>
class SettingsInterpolationUser : public ISettingsInterpolationUser
{
public:
    using Transform = TransformType;
    using Settings = typename TransformType::Settings;

    std::vector<int> GetKeyframeNumbers() const override
    {
        std::vector<int> frames;
        frames.reserve(_mSettings.size());
        for (const auto& entry : _mSettings)
            frames.push_back(entry.first);
        return frames;
    }

private:
    std::map<int, Settings> _mSettings;
    PipelineElementWindow* _pHostWindow = nullptr;

protected:
    virtual Settings GetCurrentSettings() const = 0;

    SettingsInterpolationUser(PipelineElementWindow* pHostWindow, Settings defaultSettings)
    : _pHostWindow( pHostWindow ) 
    {
        _mSettings[0] = defaultSettings;
    }

    std::pair<int,Settings> GetKeyFrameForFrame(int frameIndex) const
    {
        auto it = _mSettings.upper_bound(frameIndex);
        if ( it == _mSettings.end() )
        {
            return *std::prev(it);
        }

        if ( it == _mSettings.begin() )
            return *it;

        return *std::prev(it);
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

    void DrawFrameCounter(float availableWidth) override
    {
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float buttonWidth = std::max(1.0f, (availableWidth - spacing) * 0.5f);

        int currentFrame = _pHostWindow->GetPreviewedFrameNumber();
        auto it = _mSettings.find(currentFrame);
        if ( it != _mSettings.end() )
        {
            UI::Button("Commit", { buttonWidth, 0 }, [&]
            {
                _pHostWindow->OnKeyframeCommited();
            }, "Commit this keyframe", nullptr);
            
            // Can't delete the first keyframe
            if ( currentFrame != 0 )
            {
                ImGui::SameLine();

                UI::Button("Delete", { buttonWidth, 0 }, [&]
                {
                    _mSettings.erase(it);
                }, "Delete this keyframe", nullptr);
            }
        }
        else
        {
            UI::Button("Add", { availableWidth, 0 }, [&]
            {
                _pHostWindow->OnKeyframeCommited();
            }, "Add this keyframe", nullptr );
        }

        UI::Button("< Prev", { buttonWidth, 0 },
                   [&]
        {
            auto it = _mSettings.lower_bound(currentFrame);
            if ( it != _mSettings.begin() )
                --it;

            currentFrame = it->first;
            _pHostWindow->OnPreviewedFrameNumberChanged(currentFrame);
        }, "Go to previous keyframe", nullptr);

        ImGui::SameLine();

        UI::Button("Next >", { buttonWidth, 0 },
                   [&]
        {
            auto it = _mSettings.upper_bound(currentFrame);
            if ( it != _mSettings.end() )
            {
                currentFrame = it->first;
                _pHostWindow->OnPreviewedFrameNumberChanged(currentFrame);
            }
        }, "Go to next keyframe", nullptr);

        ImGui::BeginDisabled(_mSettings.lower_bound(currentFrame) == _mSettings.begin());
        UI::Button("Assign settings to\nprev keyframe", { availableWidth, 0 }, [&]
        {
            auto previous = _mSettings.lower_bound(currentFrame);
            if (previous != _mSettings.begin())
                std::prev(previous)->second = GetCurrentSettings();
        }, "Assign current settings to the previous keyframe", _pHostWindow);
        ImGui::EndDisabled();
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

    void CutExtraKeyframes(int maxFrame) override
    {
        for ( auto it = _mSettings.lower_bound( maxFrame + 1 ); it != _mSettings.end(); )
        {
            it = _mSettings.erase( it );
        }
    }
};

ACMB_GUI_NAMESPACE_END

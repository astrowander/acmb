#pragma once

#include "../Core/macros.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

ACMB_GUI_NAMESPACE_BEGIN

struct LayoutPreferences
{
    inline static constexpr float MinimumToolboxWidth = 220;
    inline static constexpr float MinimumInspectorWidth = 260;
    inline static constexpr float MaximumPanelWidth = 800;
    inline static constexpr float MinimumPipelineHeight = 160;
    inline static constexpr float MaximumPipelineHeight = 600;
    float toolboxWidth = 320;
    float inspectorWidth = 340;
    float pipelineHeight = 160;
    bool pipelineCollapsed = false;

    bool operator==(const LayoutPreferences&) const = default;

    void Clamp()
    {
        toolboxWidth = std::isfinite(toolboxWidth) ? std::clamp(toolboxWidth, MinimumToolboxWidth, MaximumPanelWidth) : 320;
        inspectorWidth = std::isfinite(inspectorWidth) ? std::clamp(inspectorWidth, MinimumInspectorWidth, MaximumPanelWidth) : 340;
        pipelineHeight = std::isfinite(pipelineHeight) ? std::clamp(pipelineHeight, MinimumPipelineHeight, MaximumPipelineHeight) : 160;
    }

    bool Load(const std::filesystem::path& path)
    {
        std::ifstream in(path);
        LayoutPreferences loaded;
        std::string key;
        int version = 0, collapsed = 0;
        if (!(in >> key >> version) || key != "version" || version != 1 ||
            !(in >> key >> loaded.toolboxWidth) || key != "toolbox_width" ||
            !(in >> key >> loaded.inspectorWidth) || key != "inspector_width" ||
            !(in >> key >> loaded.pipelineHeight) || key != "pipeline_height" ||
            !(in >> key >> collapsed) || key != "pipeline_collapsed" || (collapsed != 0 && collapsed != 1))
            return false;
        loaded.pipelineCollapsed = collapsed != 0;
        loaded.Clamp();
        *this = loaded;
        return true;
    }

    bool Save(const std::filesystem::path& path) const
    {
        try
        {
            if (!path.parent_path().empty())
                std::filesystem::create_directories(path.parent_path());
            auto temporary = path;
            temporary += ".tmp";
            auto saved = *this;
            saved.Clamp();
            std::ofstream out(temporary, std::ios::trunc);
            out << std::setprecision(std::numeric_limits<float>::max_digits10)
                << "version 1\ntoolbox_width " << saved.toolboxWidth
                << "\ninspector_width " << saved.inspectorWidth
                << "\npipeline_height " << saved.pipelineHeight
                << "\npipeline_collapsed " << int(saved.pipelineCollapsed) << '\n';
            out.flush();
            if (!out) return false;
            out.close();
            std::filesystem::rename(temporary, path);
            return true;
        }
        catch (const std::exception&) { return false; }
    }
};

ACMB_GUI_NAMESPACE_END

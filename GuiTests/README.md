# acmb-gui-tests

Console regression-test target for GUI code. Reuses `Tests/test.h` and
`Tests/TestRunner.cpp`, with independent suite registration and a nonzero exit
code on failure. It does not run the acmb-lib test suites.

## Build and run on Windows

From a Visual Studio developer command prompt at the repository root:

```bat
msbuild GuiTests\acmb-gui-tests.vcxproj /p:Configuration=Debug /p:Platform=x64
GuiTests\x64\Debug\acmb-gui-tests.exe
```

The target is also included in `AstroCombiner.sln` and `GUI/acmb-gui.sln`.
Its output paths are relative to the project directory, including when built
through either solution.

## Build and run with qmake

```sh
mkdir -p build-gui-tests
cd build-gui-tests
qmake ../GuiTests/acmb-gui-tests.pro
make
./acmb-gui-tests
```

## Adding tests

Use the existing `BEGIN_SUITE`, `BEGIN_TEST`, and `EXPECT_*` conventions and
add each source file to both project files. No additional test framework is
required.

The initial suites verify headless ImGui frame generation and GUI menu callback
arguments. They use the GUI's vendored ImGui core without a platform or renderer
backend, CUDA, or image-processing dependencies. The context disables ImGui
settings and log files so tests do not touch user preferences.

This target is a starting point for GUI regression coverage. It does not yet
instantiate `MainWindow`, automate application workflows, or compare rendered
screenshots; those need further application harness integration.

## Deterministic layout inputs

`TestAppHost` owns an isolated headless context and accepts `LayoutInputs`:

```cpp
TestAppHost host({{1366, 768}, 1.25f, {1.0f, 1.0f}});
host.RenderFrame([] { /* draw the component under test */ });
host.Resize({1280, 720});
```

Viewport size is in logical ImGui units. UI scale changes style dimensions and
the embedded test font; framebuffer scale controls pixel density independently.
The host uses a fixed frame interval, a fresh style, no settings/log files, and
no monitor queries. Destruction restores the previously active context.

`LayoutInputs` covers the size/scale matrix, repeatable geometry, scale effects,
framebuffer density, resize/recovery, invalid inputs, and context isolation.
These tests validate the test host and ImGui layout inputs. Native Windows DPI
handling and production fonts require separate integration coverage.

## MainWindow panel geometry

`MainWindowPanels` exercises `GUI/MainWindowLayout.h`, the production geometry
calculator used by `MainWindow` for its toolbox, preview, frame controls,
pipeline, preview sizing, and pipeline mouse bounds. It runs without constructing
the application singleton or loading CUDA and image-processing dependencies.

Coverage includes a fixed desktop reference, panel containment and overlap across
four viewport sizes at 100%, 125%, and 150% UI scale, toolbox/pipeline separation,
pipeline column thresholds, small/minimized windows, resize/restore, and
framebuffer-density independence. Panel containment is checked against the
workspace canvas. The canvas has a minimum size and scrolls within smaller
windows; minimized windows have empty panel bounds.

These are production geometry regression tests, not screenshot or complete
`MainWindow` rendering/interaction tests. They do not establish that tool buttons,
frame widgets, or pipeline cards fit inside every panel.

## Pipeline editing

`PipelineEditing` exercises the shared production algorithms in
`GUI/PipelineEditing.h`. `MainWindow` uses these algorithms for tool insertion
and replacement, Delete, Clear, arrow-key selection, and selection visibility.

The 12 tests cover empty/head/middle/tail edits, replacement at the end,
invalid positions, links in both directions, successor input notifications,
node lifetime, clearing a scrolled pipeline, keyboard navigation, and selection
after deletion or a narrower viewport. Positions beyond the insertion slot at
the end are rejected; deleting a nonexistent element is a no-op.

Tests substitute lightweight nodes with the same strong-output/weak-input link
API. They verify the production editing rules without loading processing tools.
They do not yet inject ImGui keyboard/mouse events or verify tool-specific
preview invalidation and `OnInputChanged` behavior.

## Input-driven interaction tests

`GuiInteractions` adds three component interaction tests. `TestAppHost::Click`
and `PressKey` inject mouse and keyboard events through ImGui IO and advance
frames; tests do not invoke control callbacks directly.

- Selection: double-click a pipeline cell using production panel geometry,
  then navigate steps and insertion positions with the arrow keys.
- Settings: click radio options, verify only the selected step changes, and
  verify a locked setting rejects changes.
- Run/Stop: click Run, verify repeated Run and edits are blocked while busy,
  click Stop, and verify cancellation keeps edits locked until worker completion.
  Verify edits and a fresh Run work after completion.

`GUI/GuiInteractions.h` contains the input handlers and processing state used by
`MainWindow` and `ImGuiHelpers`. The tests use those same production components
with small fixture windows. Button coordinates come from the rendered item
bounds, and processing completion uses a promise instead of sleeps or real
image processing.

These are headless component interactions, not full application workflows.
Application startup, autosave, menu registration, tool-specific preview updates,
and the real export worker still need integration coverage. No additional test
engine dependency is needed for these three cases.

## Minimum sizes and scrolling

The workspace preserves a minimum 1000 x 600 logical canvas (scaled for larger
fonts). Pipeline cards start at 280 x 250 logical units, with independent vertical
settings scrolling. The pipeline scrolls horizontally instead of compressing its
cards. Selecting a step with the keyboard brings it into view. Toolbox commands
are single icon-and-text rows using the merged icon/text font, with no manually
positioned captions. Frame/keyframe settings scroll vertically and button pairs
wrap onto separate rows when they do not fit.

`ScrollablePanels` verifies minimum canvas/panel/card dimensions, non-overlapping
icon-and-text rows, original theme sizing for settings controls, scrollbars on small workspaces,
wheel scrolling to the final tool/setting, fixed card sizes after viewport
shrinkage, and selection hit testing after horizontal scrolling. These checks use
the same child-panel helpers as the production GUI.

Pipeline cards retain visible borders (red for the selected card) and centered
title bars and column captions. Each title bar uses the original window name and
title-background style, spans the card width, and stays visible while its settings
scroll independently below it. Settings controls use the original frame padding,
natural button heights, and compact numeric widths; scrolling provides access
to overflowing controls without enlarging them. The pipeline also draws an outer
grid border, column dividers, and a header separator. Cards are centered inside
their cells with horizontal and vertical gutters; their settings viewports respect
the card's window padding, including space around the scrollbar.

Clear Pipeline and the interaction-help tooltip live in a separate fixed column
to the left of the scrolling cards. A horizontal separator runs above the pipeline
area, and a vertical divider separates its command column from the card grid.

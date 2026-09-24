# MemPulse

These Windows x64 binaries and the public header come from [MemPulse](https://github.com/jpola-amd/mempulse), version 1.0.4, commit `24506d0944380825fea13cea83e141ada2a9a125`. The DLL and import library were built in Release mode with MSVC 19.51 and HIP 7.14. The upstream MIT license is included in `LICENSE`.

The VMM tiled loading example always links against these files. The bundled binaries support Windows x64. CMake copies `bin/x64/mempulse.dll` beside the example executable and installs it in `bin`.

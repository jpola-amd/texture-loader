#pragma once

#include <cstdarg>

namespace hip_demand {

enum class LogLevel : int {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4
};

// Set global log level (default Off).
void setLogLevel(LogLevel level);

// Get current global log level.
LogLevel getLogLevel();

// printf-style logger (host-side only); no-op when level is above current threshold.
void logMessage(LogLevel level, const char* fmt, ...);

} // namespace hip_demand

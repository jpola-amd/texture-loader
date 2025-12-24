#include "DemandLoading/Logging.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

namespace hip_demand {
namespace {
std::atomic<LogLevel> gLogLevel{LogLevel::Off};
std::mutex gLogMutex;

const char* levelTag(LogLevel level) {
    switch (level) {
        case LogLevel::Error: return "[error] ";
        case LogLevel::Warn:  return "[warn ] ";
        case LogLevel::Info:  return "[info ] ";
        case LogLevel::Debug: return "[debug] ";
        default: return "";
    }
}
} // namespace

void setLogLevel(LogLevel level) {
    gLogLevel.store(level, std::memory_order_relaxed);
}

LogLevel getLogLevel() {
    return gLogLevel.load(std::memory_order_relaxed);
}

void logMessage(LogLevel level, const char* fmt, ...) {
    if (level == LogLevel::Off) {
        return;
    }
    LogLevel current = gLogLevel.load(std::memory_order_relaxed);
    if (static_cast<int>(level) > static_cast<int>(current)) {
        return;
    }

    std::lock_guard<std::mutex> lock(gLogMutex);
    std::fputs(levelTag(level), stderr);

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    if (fmt && fmt[0] != '\0') {
        char last = fmt[std::char_traits<char>::length(fmt) - 1];
        if (last != '\n') {
            std::fputc('\n', stderr);
        }
    }
    std::fflush(stderr);
}

} // namespace hip_demand

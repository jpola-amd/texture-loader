#include "DemandLoading/Logging.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace hip_demand {
namespace {

std::atomic<LogLevel> gLogLevel{LogLevel::Off};

// Use a spinlock for minimal contention on short log messages
// This avoids heavy mutex overhead in hot paths
class SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
public:
    void lock() noexcept {
        // Use exponential backoff to reduce cache-line contention
        for (int spin = 0; flag_.test_and_set(std::memory_order_acquire); ++spin) {
            if (spin < 16) {
                // Busy-wait for a few iterations
#if defined(_MSC_VER)
                _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#else
                // ARM or other: just yield
                std::this_thread::yield();
#endif
            } else {
                // After spinning, yield to OS scheduler
                std::this_thread::yield();
            }
        }
    }
    
    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }
};

SpinLock gLogLock;

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

    // Format message into thread-local buffer to minimize lock hold time
    thread_local char buffer[2048];
    
    va_list args;
    va_start(args, fmt);
    int len = std::vsnprintf(buffer, sizeof(buffer) - 1, fmt, args);
    va_end(args);
    
    if (len < 0) len = 0;
    buffer[sizeof(buffer) - 1] = '\0';
    
    // Add newline if not present
    bool needsNewline = (len > 0 && buffer[len - 1] != '\n');
    
    // Only hold lock during actual I/O
    gLogLock.lock();
    std::fputs(levelTag(level), stderr);
    std::fputs(buffer, stderr);
    if (needsNewline) {
        std::fputc('\n', stderr);
    }
    std::fflush(stderr);
    gLogLock.unlock();
}

} // namespace hip_demand

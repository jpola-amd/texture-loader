#pragma once

// Platform detection
#ifdef _WIN32
	#define MEMPULSE_PLATFORM_WINDOWS 1
	#define MEMPULSE_PLATFORM_LINUX 0
#else
	#define MEMPULSE_PLATFORM_WINDOWS 0
	#define MEMPULSE_PLATFORM_LINUX 1
#endif // API export/import macros for Windows

#ifdef _WIN32
	#ifdef MEMPULSE_EXPORT_API
		#define MEMPULSE_API __declspec(dllexport)
	#else
		#define MEMPULSE_API __declspec(dllimport)
    #endif
#else
    #define MEMPULSE_API __attribute__((visibility("default")))
#endif

// Compiler-specific attributes
#ifdef __GNUC__
    #define MEMPULSE_DEPRECATED __attribute__((deprecated))
    #define MEMPULSE_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
    #define MEMPULSE_DEPRECATED __declspec(deprecated)
    #define MEMPULSE_UNUSED __pragma(warning(suppress:4100))
#else
    #define MEMPULSE_DEPRECATED
    #define MEMPULSE_UNUSED
#endif

#ifdef NDEBUG
    #define MEMPULSE_DEBUG 0
#else
    #define MEMPULSE_DEBUG 1
#endif

#if MEMPULSE_PLATFORM_WINDOWS
    #define MEMPULSE_HAS_D3DKMT 1
    #define MEMPULSE_HAS_HIP_MEMORY_BUDGET 0
#else
    #define MEMPULSE_HAS_D3DKMT 0
    #define MEMPULSE_HAS_HIP_MEMORY_BUDGET 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MempulseError {
    MEMPULSE_SUCCESS = 0,
    MEMPULSE_ERROR_INTERNAL = -1,
	MEMPULSE_ERROR_INVALID_PARAMETER = -2,
	MEMPULSE_ERROR_INVALID_CONTEXT = -3,
	MEMPULSE_ERROR_INVALID_MEMORY = -4,
	MEMPULSE_ERROR_INVALID_DEVICE = -5,
	MEMPULSE_ERROR_HIP = -6,
	MEMPULSE_ERROR_DRM = -7,
	MEMPULSE_ERROR_D3DKMT = -8,
} MempulseError;

typedef enum MempulseBackend {
    MEMPULSE_BACKEND_ANY = 0,
    MEMPULSE_BACKEND_DRM = 1,
    MEMPULSE_BACKEND_HIP = 2,
    MEMPULSE_BACKEND_D3DKMT = 3,
    MEMPULSE_BACKEND_END = 4,
} MempulseBackend;


typedef struct MempulseDeviceMemoryInfo {
    // on linux only dedicated is reported
    // for integrated gpus it account the memory in dedicated section
    unsigned long long int dedicatedUsed;
    unsigned long long int dedicatedTotal;
    unsigned long long int sharedUsed;
    unsigned long long int sharedTotal;
} MempulseDeviceMemoryInfo;

typedef struct MempulseDeviceMemoryUsage {
	unsigned long long free;
	unsigned long long total;
} MempulseDeviceMemoryUsage;


typedef void* MempulseContext;

MEMPULSE_API MempulseError MempulseInitialize(MempulseContext* context, MempulseBackend backend = MEMPULSE_BACKEND_ANY);

MEMPULSE_API MempulseError MempulseShutdown(MempulseContext context);

MEMPULSE_API MempulseError MempulseSetLoggingLevel(int level);

MEMPULSE_API const char* MempulseGetVersion();

MEMPULSE_API MempulseError MempulseGetAvailabeDeviceCount(
		MempulseContext context,
		int* count);

MEMPULSE_API MempulseError MempulseGetDeviceName(
		MempulseContext context,
		int deviceIndex,
		char* deviceName,
		unsigned long long int deviceNameSize);

MEMPULSE_API MempulseError MempulseGetDeviceMemoryInfo(
		MempulseContext context,
		int deviceIndex,
		MempulseDeviceMemoryInfo* deviceMemoryInfo);

MEMPULSE_API MempulseError MempulseGetDeviceMemoryUsage(
		MempulseContext context,
		int deviceIndex,
		MempulseDeviceMemoryUsage* deviceMemoryUsage);

MEMPULSE_API MempulseError MempulsePrintDeviceMemoryInfo(
		const MempulseDeviceMemoryInfo* deviceMemoryInfo);

MEMPULSE_API MempulseError MempulsePrintDeviceMemoryUsage(
		const MempulseDeviceMemoryUsage* deviceMemoryUsage);

#ifdef __cplusplus
} // extern "C"
#endif

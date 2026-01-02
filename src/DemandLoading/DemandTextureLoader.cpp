// SPDX-License-Identifier: MIT
// DemandTextureLoader public API implementation

#include <hip/hip_runtime.h>
#include <DemandLoading/DemandTextureLoader.h>
#include "DemandTextureLoaderImpl.h"
namespace hip_demand {

// -----------------------------------------------------------------------------
// Error String Conversion
// -----------------------------------------------------------------------------

const char* getErrorString(LoaderError error) {
    switch (error) {
        case LoaderError::Success:            return "Success";
        case LoaderError::InvalidTextureId:   return "Invalid texture ID";
        case LoaderError::MaxTexturesExceeded: return "Maximum textures exceeded";
        case LoaderError::FileNotFound:       return "File not found";
        case LoaderError::ImageLoadFailed:    return "Image load failed";
        case LoaderError::OutOfMemory:        return "Out of memory";
        case LoaderError::InvalidParameter:   return "Invalid parameter";
        case LoaderError::HipError:           return "HIP error";
        default:                              return "Unknown error";
    }
}

// -----------------------------------------------------------------------------
// Public API - Forward to Impl
// -----------------------------------------------------------------------------

DemandTextureLoader::DemandTextureLoader(const LoaderOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}

DemandTextureLoader::~DemandTextureLoader() = default;

TextureHandle DemandTextureLoader::createTexture(const std::string& filename,
                                                 const TextureDesc& desc) {
    return impl_->createTexture(filename, desc);
}

TextureHandle DemandTextureLoader::createTexture(std::shared_ptr<ImageSource> imageSource,
                                                 const TextureDesc& desc) {
    return impl_->createTexture(std::move(imageSource), desc);
}

TextureHandle DemandTextureLoader::createTextureFromMemory(const void* data,
                                                           int width, int height, int channels,
                                                           const TextureDesc& desc) {
    return impl_->createTextureFromMemory(data, width, height, channels, desc);
}

void DemandTextureLoader::launchPrepare(hipStream_t stream) {
    impl_->launchPrepare(stream);
}

DeviceContext DemandTextureLoader::getDeviceContext() const {
    return impl_->getDeviceContext();
}

size_t DemandTextureLoader::processRequests(hipStream_t stream, const DeviceContext& deviceContext) {
    return impl_->processRequests(stream, deviceContext);
}

Ticket DemandTextureLoader::processRequestsAsync(hipStream_t stream, const DeviceContext& deviceContext) {
    return impl_->processRequestsAsync(stream, deviceContext);
}

size_t DemandTextureLoader::getResidentTextureCount() const {
    return impl_->getResidentTextureCount();
}

size_t DemandTextureLoader::getTotalTextureMemory() const {
    return impl_->getTotalTextureMemory();
}

size_t DemandTextureLoader::getRequestCount() const {
    return impl_->getRequestCount();
}

bool DemandTextureLoader::hadRequestOverflow() const {
    return impl_->hadRequestOverflow();
}

LoaderError DemandTextureLoader::getLastError() const {
    return impl_->getLastError();
}

void DemandTextureLoader::enableEviction(bool enable) {
    impl_->enableEviction(enable);
}

void DemandTextureLoader::setMaxTextureMemory(size_t bytes) {
    impl_->setMaxTextureMemory(bytes);
}

size_t DemandTextureLoader::getMaxTextureMemory() const {
    return impl_->getMaxTextureMemory();
}

void DemandTextureLoader::updateEvictionPriority(uint32_t textureId, EvictionPriority priority) {
    impl_->updateEvictionPriority(textureId, priority);
}

void DemandTextureLoader::unloadTexture(uint32_t textureId) {
    impl_->unloadTexture(textureId);
}

void DemandTextureLoader::unloadAll() {
    impl_->unloadAll();
}

void DemandTextureLoader::abort() {
    impl_->abort();
}

bool DemandTextureLoader::isAborted() const {
    return impl_->isAborted();
}

} // namespace hip_demand

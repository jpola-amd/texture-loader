// Example demonstrating OpenImageIO image format support
// Loads and compares various image formats (EXR, HDR, TIFF, PNG)
// Shows format detection, statistics, and pixel data access

#include "DemandLoading/DemandTextureLoader.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#ifdef USE_OIIO
#include "ImageSource/ImageSource.h"
#include "ImageSource/OIIOReader.h"
#include "ImageSource/TextureInfo.h"
#endif

void printSeparator() {
    std::cout << std::string(80, '=') << std::endl;
}

void testImageFormat(const std::string& filename) {
    std::cout << "\nTesting: " << filename << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
#ifdef USE_OIIO
    try {
        auto imgSrc = hip_demand::createImageSource(filename);
        if (!imgSrc) {
            std::cout << "ERROR: Failed to create ImageSource" << std::endl;
            return;
        }
        
        hip_demand::TextureInfo info;
        imgSrc->open(&info);
        
        if (!info.isValid) {
            std::cout << "ERROR: Failed to open file or invalid texture" << std::endl;
            return;
        }
        
        // Print image information
        std::cout << "Resolution:    " << info.width << " x " << info.height << std::endl;
        std::cout << "Channels:      " << info.numChannels << std::endl;
        std::cout << "Mip Levels:    " << info.numMipLevels << std::endl;
        std::cout << "Pixel Format:  ";
        
        switch (info.format) {
            case hip_demand::PixelFormat::UINT8:   std::cout << "UINT8 (8-bit)"; break;
            case hip_demand::PixelFormat::UINT16:  std::cout << "UINT16 (16-bit)"; break;
            case hip_demand::PixelFormat::FLOAT16: std::cout << "FLOAT16 (half)"; break;
            case hip_demand::PixelFormat::FLOAT32: std::cout << "FLOAT32 (float)"; break;
            default: std::cout << "Unknown"; break;
        }
        std::cout << std::endl;
        
        // Calculate sizes
        size_t bytesPerPixel = hip_demand::getBytesPerChannel(info.format) * info.numChannels;
        size_t baseSize = info.width * info.height * bytesPerPixel;
        size_t totalSize = hip_demand::getTextureSizeInBytes(info);
        
        std::cout << "Base Size:     " << baseSize / 1024 << " KB" << std::endl;
        std::cout << "Total w/mips:  " << totalSize / 1024 << " KB" << std::endl;
        
        // Read base level pixel data
        std::vector<char> buffer(baseSize);
        if (imgSrc->readMipLevel(buffer.data(), 0, info.width, info.height)) {
            std::cout << "[OK] Successfully read base mip level" << std::endl;
            
            // Sample center pixel (assuming UINT8 for display)
            if (info.format == hip_demand::PixelFormat::UINT8 && info.numChannels >= 3) {
                size_t centerPixel = (info.height / 2) * info.width + (info.width / 2);
                size_t pixelOffset = centerPixel * info.numChannels;
                
                std::cout << "Center pixel:  RGB(" 
                          << (int)(unsigned char)buffer[pixelOffset + 0] << ", "
                          << (int)(unsigned char)buffer[pixelOffset + 1] << ", "
                          << (int)(unsigned char)buffer[pixelOffset + 2];
                if (info.numChannels >= 4) {
                    std::cout << ", " << (int)(unsigned char)buffer[pixelOffset + 3];
                }
                std::cout << ")" << std::endl;
            }
        } else {
            std::cout << "[ERROR] Failed to read pixel data" << std::endl;
        }
        
        // Read base color
        float4 baseColor;
        if (imgSrc->readBaseColor(baseColor)) {
            std::cout << "Base Color:    RGB(" 
                      << std::fixed << std::setprecision(3)
                      << baseColor.x << ", "
                      << baseColor.y << ", "
                      << baseColor.z << ")" << std::endl;
        }
        
        // Statistics
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "  Bytes Read:  " << imgSrc->getNumBytesRead() / 1024 << " KB" << std::endl;
        std::cout << "  Read Time:   " << std::fixed << std::setprecision(3) 
                  << imgSrc->getTotalReadTime() * 1000.0 << " ms" << std::endl;
        
        imgSrc->close();
        std::cout << "[OK] Successfully closed" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "UNKNOWN EXCEPTION" << std::endl;
    }
#else
    std::cout << "ERROR: Compiled without OpenImageIO support" << std::endl;
    std::cout << "       Rebuild with -DUSE_OIIO=ON" << std::endl;
#endif
}

void testWithDemandLoader(const std::string& filename) {
    std::cout << "\nTesting with DemandTextureLoader: " << filename << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    hip_demand::LoaderOptions options;
    options.maxTextureMemory = 512 * 1024 * 1024;  // 512 MB
    options.maxTextures = 100;
    options.enableEviction = false;
    
    hip_demand::DemandTextureLoader loader(options);
    
    hip_demand::TextureDesc desc;
    desc.generateMipmaps = true;
    desc.filterMode = hipFilterModeLinear;
    
    auto handle = loader.createTexture(filename, desc);
    
    if (handle.valid) {
        std::cout << "[OK] Texture created successfully" << std::endl;
        std::cout << "  Texture ID:  " << handle.id << std::endl;
        std::cout << "  Resolution:  " << handle.width << " x " << handle.height << std::endl;
        std::cout << "  Channels:    " << handle.channels << std::endl;
    } else {
        std::cout << "[ERROR] Failed to create texture: " 
                  << hip_demand::getErrorString(handle.error) << std::endl;
    }
}

int main(int argc, char** argv) {
    printSeparator();
    std::cout << "OpenImageIO Format Support Example" << std::endl;
    std::cout << "HIP Demand Texture Loader" << std::endl;
    printSeparator();
    
#ifdef USE_OIIO
    std::cout << "\n[OK] OpenImageIO support ENABLED" << std::endl;
#else
    std::cout << "\n[ERROR] OpenImageIO support DISABLED" << std::endl;
    std::cout << "  Rebuild with -DUSE_OIIO=ON to enable advanced format support" << std::endl;
    return 1;
#endif
    
    // Test various image formats
    std::vector<std::string> testFiles;
    
    if (argc > 1) {
        // Use files provided as arguments
        for (int i = 1; i < argc; ++i) {
            testFiles.push_back(argv[i]);
        }
    } else {
        // Default test files (user should provide these)
        std::cout << "\nUsage: " << argv[0] << " <image1> [image2] [image3] ..." << std::endl;
        std::cout << "\nRecommended test files:" << std::endl;
        std::cout << "  - test.exr      (OpenEXR HDR format)" << std::endl;
        std::cout << "  - test.hdr      (Radiance HDR format)" << std::endl;
        std::cout << "  - test_16.tif   (16-bit TIFF)" << std::endl;
        std::cout << "  - test.png      (Standard PNG)" << std::endl;
        std::cout << "  - test.jpg      (JPEG)" << std::endl;
        
        // Try to find some default test files
        testFiles = {
            "test.exr",
            "test.hdr",
            "test.tif",
            "test.png",
            "test.jpg"
        };
        
        std::cout << "\nAttempting to load default test files..." << std::endl;
    }
    
    // Test each file with direct OIIO access
    printSeparator();
    std::cout << "PART 1: Direct ImageSource Testing" << std::endl;
    printSeparator();
    
    for (const auto& file : testFiles) {
        testImageFormat(file);
    }
    
    // Test with DemandTextureLoader
    printSeparator();
    std::cout << "\nPART 2: DemandTextureLoader Integration" << std::endl;
    printSeparator();
    
    for (const auto& file : testFiles) {
        testWithDemandLoader(file);
    }
    
    printSeparator();
    std::cout << "\nFormat Support Summary:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "With OpenImageIO enabled, the following formats are supported:" << std::endl;
    std::cout << "  [+] EXR       - OpenEXR (16/32-bit float, HDR)" << std::endl;
    std::cout << "  [+] HDR       - Radiance HDR (32-bit float)" << std::endl;
    std::cout << "  [+] TIFF      - Tagged Image File Format (8/16/32-bit)" << std::endl;
    std::cout << "  [+] PNG       - Portable Network Graphics (8/16-bit)" << std::endl;
    std::cout << "  [+] JPEG      - JPEG/JFIF (8-bit)" << std::endl;
    std::cout << "  [+] TGA       - Truevision Targa" << std::endl;
    std::cout << "  [+] BMP       - Windows Bitmap" << std::endl;
    std::cout << "  [+] DPX       - Digital Picture Exchange" << std::endl;
    std::cout << "  [+] And 100+ more via OIIO plugins" << std::endl;
    std::cout << "\nAll formats are automatically converted to UINT8 RGBA for GPU upload." << std::endl;
    printSeparator();
    
    return 0;
}

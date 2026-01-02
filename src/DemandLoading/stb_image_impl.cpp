// Separate compilation unit for stb_image to avoid multiple definition issues
// when the library is linked multiple times

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

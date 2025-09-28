cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(ScapularMorphology)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/liukuan5625/SlicerScapularMorphology")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Kuan Liu (Ruijin Hospital, Shanghai Jiaotong University School of Medicine)")
set(EXTENSION_DESCRIPTION "Scapular Morphology for glenoid segmentation and landmarks location.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/liukuan5625/SlicerScapularMorphology/main/ScapularMorphology.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/liukuan5625/SlicerScapularMorphology/main/Screenshot01.png")
set(EXTENSION_DEPENDS "PyTorch")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ScapularMorphology)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

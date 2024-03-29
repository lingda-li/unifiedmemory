add_library(CudaPass MODULE
    # List your source files here.
    CudaPass.cpp
)

# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(CudaPass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(CudaPass PROPERTIES
    COMPILE_FLAGS "-D__GLIBCXX_USE_CXX11_ABI=0 -fno-rtti"
)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(CudaPass PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)

add_library(CudaModulePass MODULE
    # List your source files here.
    CudaModulePass.cpp
)

# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(CudaModulePass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(CudaModulePass PROPERTIES
    COMPILE_FLAGS "-D__GLIBCXX_USE_CXX11_ABI=0 -fno-rtti"
)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(CudaModulePass PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)

add_library(UVMTransformPass MODULE
    # List your source files here.
    UVMTransformPass.cpp
    MemAccessInfo.cpp
    CGMemAnalysis.cpp
)

# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(UVMTransformPass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(UVMTransformPass PROPERTIES
    COMPILE_FLAGS "-D__GLIBCXX_USE_CXX11_ABI=0 -fno-rtti"
)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(UVMTransformPass PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)

add_library(MemAnalysisPass MODULE
    # List your source files here.
    MemAnalysisPass.cpp
    MemAccessInfo.cpp
    CGMemAnalysis.cpp
)

# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(MemAnalysisPass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(MemAnalysisPass PROPERTIES
    COMPILE_FLAGS "-D__GLIBCXX_USE_CXX11_ABI=0 -fno-rtti"
)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(MemAnalysisPass PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)

add_library(OMPPass MODULE
    # List your source files here.
    OMPPass.cpp
    CGMemAnalysis.cpp
    TFGPass.cpp
)

# Use C++11 to compile our pass (i.e., supply -std=c++11).
target_compile_features(OMPPass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is (typically) built with no C++ RTTI. We need to match that;
# otherwise, we'll get linker errors about missing RTTI data.
set_target_properties(OMPPass PROPERTIES
    COMPILE_FLAGS "-D__GLIBCXX_USE_CXX11_ABI=0 -fno-rtti"
)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
    set_target_properties(OMPPass PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)

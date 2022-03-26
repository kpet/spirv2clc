# spirv2clc [![CI badge](https://github.com/kpet/spirv2clc/actions/workflows/presubmit.yml/badge.svg?branch=main)](https://github.com/kpet/spirv2clc/actions/workflows/presubmit.yml?query=branch%3Amain++)

spirv2clc is an experimental OpenCL SPIR-V to OpenCL C translator currently
targeting OpenCL 1.2 support. It can generate OpenCL C code equivalent to an
input OpenCL SPIR-V module. The generated code is not meant to be human-readable
and closely follows the input SPIR-V. It is intended to be used as a means for
OpenCL applications that require SPIR-V to run on OpenCL implementations that
only support OpenCL C.

# Dependencies

This project depends on the following other projects:

* [OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers)
* [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers)
* [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)

# Building

spirv2clc uses CMake as its build system and the following should work on most systems:

```
git submodule update --init
mkdir build
cd build
cmake ..
cmake --build .
```

# Using the translator tool

A command line translator tool is provided and can be used as follows:

```
./build/tools/spirv2clc module.spv
```

The translated source is printed to the standard output.

The tool supports the following options:

- `--asm` treat the input as SPIR-V assembly in text form.

# Embedding as a library

spirv2clc can be embedded as a library and used as follows:

```
#include "spirv2clc.h"

spirv2clc::translator translator;
std::string srcgen;
std::vector<uint32_t> binary;
int err = translator.translate(binary, &srcgen);
```

## Installation

To install the library, first check you're building with the right CMake variables set:
* `CMAKE_INSTALL_PREFIX` - Location of the final package.
* `BUILD_SHARED_LIBS` - Whether to build spirv2clc as a shared library or a static library.
  Note that a static library build at the moment requires the user
  to manually provide the SPIR-V dependencies as static libraries
  in the target project (where you want to embed spirv2clc).

To install the library after a successful build, run the following command:
```sh
cmake --install .
```

After the installation step you can use spirv2clc as an `IMPORTED` library
in your target project.

# Running with test layer

A layer that enables a round-trip translation of OpenCL C programs to SPIR-V and
back to OpenCL C is provided for testing purposes. It intercepts OpenCL calls
performed by applications and calls offline tools to handle program substitutions.
The flow is as follows:

- Intercept program creation OpenCL calls
- Save sources
- Compile to LLVM IR (using `clang`)
- Translate LLVM IR to SPIR-V (using `llvm-spirv`)
- Translate back to OpenCL C (using `spirv2clc`)

The layer expects the following tools to be in the `PATH`:

- `clang`
- `llvm-spirv`
- `spirv2clc`

and can be used as follows:

```
LD_PRELOAD=./build/testlayer/libtestlayer.so \
PATH=/path/to/build:/path/to/other-tools:$PATH \
/path/to/opencl-application
```

# Design

spirv2clc uses SPIRV-Tools's IR and type analysis utilities and provides a
single-pass (with _some_ additional searching) translation of the input module.

The code generation process is rather straightforward and the generated code
closely follows the input SPIR-V with a few notable exceptions:

- `OpPhi`'s are analysed at the beginning of each function's translation to
record a full list of all of `OpPhi`'s used by the function as well as all the
variables and parents they can take as input. A variable is declared for each
`OpPhi` at the beginning of each translated function and assigned at the end of
each basic block that can be a parent to the `OpPhi`.

- Kernel scope variables in the local address space are global variables in
SPIR-V modules. At the beginning of each kernel function's translation, its
static call tree is inspected to build a list of all the variables in the
local address space it's using. A declaration is generated at the beginning of
the kernel function's source for each variable.

- Pointer to arrays are generated as pointers to array elements.

- Built-in variables are handled in the translationg of their uses.

- Booleans are translated to an appropriate (vector) integer type according to
the rules for their producer (see §6.5.4 as an example for relational operators).

All control flow is translated to `goto`'s.

# Known limitations

- No support for images
- No support for relaxed atomics
- Nesting of arrays and structures probably needs more work
- OpCompositeExtract doesn't support multiple indices

# FIDESlib32bits

A fork of [FIDESlib](https://github.com/CAPS-UMU/FIDESlib) that adds a **32-bit composite-scaling backend** and
**seed-expanded key-switching keys**. It is the runtime of
[Perseus](https://github.com/gladia-research-group/perseus) (*Perseus: Perseus: Faster FHE Transformer Inference via Complex-Packing and Sparse Bootstraps*); `CHANGES.md` lists what differs from upstream.

* `NATIVEINT=32` chains: each CKKS level is a pair of 27-bit primes (composite degree 2), so
  modular arithmetic rides native 32-bit GPU paths; the 64-bit backend stays available.
* Key-switching keys: the uniform `a` half is regenerated in-kernel from a 256-bit ChaCha12 seed
  (`FIDESLIB_KSK_REGEN`, default 2) and the `b` half is stored as a dense 28-bit bit-stream
  (`FIDESLIB_KSK_PACK`, default 1); on GPT-2 decode the two free 9.1 GiB of device memory and
  20% of end-to-end latency. `FIDESLIB_KSK_REGEN=0` / `FIDESLIB_KSK_PACK=0` restore stored /
  unpacked keys for an ablation.
* Sparse bootstraps (`EvalBootstrapSetup` with a slot count), a host staging layer for
  streaming plaintext weights through pinned memory, and a KV-cache arena for generative decode.

Measured on one RTX PRO 6000 Blackwell at logN = 16 (32-bit chain / 64-bit reference, µs):
add 33 / 29, mult ct×pt 72 / 49, mult ct×ct 203 / 243, rotate 214 / 245, bootstrap dense
28770 / 36690, sparse-512 18590 / 23590, sparse-1 16100 / 19950.

The 32-bit OpenFHE this fork builds against is a fixed upstream commit plus the patch series
kept in the Perseus repository (`third_party/openfhe-n32/patches`); Perseus's
`scripts/install_deps.sh` builds both. The `deps/` patches here are the 64-bit path.

## Citation

If you use FIDESlib on your research, please cite their ISPASS paper.

```bibtex
@inproceedings{FIDESlib,
  author    = {Carlos Agulló-Domingo and Óscar Vera-López and Seyda Guzelhan and Lohit Daksha and Aymane El Jerari and Kaustubh Shivdikar and Rashmi Agrawal and David Kaeli and Ajay Joshi and José L. Abellán},
  title     = {{FIDESlib: A Fully-Fledged Open-Source FHE Library for Efficient CKKS on GPUs}},
  booktitle = {2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  year      = {2025},
  note      = {Poster paper},
  url       = {https://github.com/CAPS-UMU/FIDESlib},
  publisher = {IEEE},
  address = {Ghent, Belgium},
}
```

## Compilation

> [!IMPORTANT]
> Requirements:
>  -  NVIDIA CUDA version 12 or greater.
>     - Must provide a NVTX implementation.
>  -  Clang Compiler toolchain version >=17 or GCC version >=11
>  -  OpenMP development library.
>  -  CMake version 3.25.2 or greater.
>  -  (Optional) NVIDIA Collective Communications Library to enable Multi-GPU support.

### Requirements installation

For CUDA software stack, follow the official installation guides. For the remaining
dependencies, install them using the package manager or mehtod of your choice. 

On Ubuntu:
```bash 
apt install make build-essential cmake git libtbb-dev libomp-dev
```

> [!NOTE]
> CMake package on Ubuntu may be older than expected; install it using snap, pip or build from source.

### FIDESlib compilation

In order to be able to compile the project, one must follow these steps:

  - Clone this repository.
  - Generate the Makefile with CMake.
  - Build the project.

FIDESlib needs a patched version of OpenFHE in order to be able to access some internals needed for interoperability. This patched version can be automatically installed by defining FIDESLIB_INSTALL_OPENFHE=ON CMake variable. By default this variable is set OFF. Once the patched version is installed, one can disable this flag when reinstalling FIDESlib.

The build process produces the following artifacts: 
- fideslib.so: The FIDESlib library to be dynamically linked to any client application.
- fideslib-test: The test suite executable if selected.
- fideslib-bench: The benchmark suite executable if selected.
- gpu-test: A dummy executable to search for the CUDA capable devices on the machine.
- dummy: Another dummy executable.

The following options can be used with CMake to configure the build. The default value for each option is denoted in **boldface** under the **Values** column:

| CMake Option                  | Values              | Description |
|-------------------------------|---------------------|------------------------------------------------------
| `FIDESLIB_ARCH`               | **"70-real;70-virtual;80-real;86-real;89-real;90-real;90-virtual;100-real;120-real"**, string | CUDA architectures to compile for (pass one, e.g. `120-real`, for a fast build). |
| `FIDESLIB_OPENFHE_NATIVE_SIZE` | **"64"**, "32" | NATIVEINT width of the OpenFHE the library is built against (32 = the composite-scaling backend). |
| `CMAKE_BUILD_TYPE`            | **"Release"**, "Debug", "MinSizeRel", "RelWithDebInfo" | Select the compilation build type. |
| `FIDESLIB_INSTALL_PREFIX`     | **"/usr/local"**,string | Select prefix path for the installation path of FIDESlib. Relative paths are resolved from the project root directory. |
| `OPENFHE_INSTALL_PREFIX`      | **"/usr/local"**,string | Select prefix path for the installation path of OpenFHE. Relative paths are resolved from the project root directory. |
| `FIDESLIB_INSTALL_OPENFHE`    | ON / **OFF**        | Enable the installation of the patched version of OpenFHE. Needed the first time. |
| `FIDESLIB_COMPILE_TESTS`      | **ON** / OFF        | Build the tests for verifying the functionality of the project. |
| `FIDESLIB_COMPILE_BENCHMARKS` | **ON** / OFF        | Build the benchmarks executable. |

### FIDESlib Installation

Installing the library is as easy as running the following command:

```bash
cmake --build $PATH_TO_BUILD_DIR --target install -j
```

FIDESlib is currently ready to be consumed as a CMake library. The template project on the examples directory shows how to build and run a FIDESlib client application and contains examples of usage of most of the functionality provided by FIDESlib.

> [!NOTE]
> As the default installation prefix for FIDESlib is /usr/local you may need administrator priviledges. Change this using the previously mentioned configuration options.

## Usage

Check examples for projects that use FIDESlib.

## Docker

Check docker directory to obtain instructions on running FIDESlib inside a Docker environment.

## Credits

Thanks to the original contributors:
* Carlos Agulló Domingo. 
* Óscar Vera López.
* Seyda Guzelhan.
* Lohit Daksha.
* Aymane El Jerari.

And thanks to their advisor:
* José L. Abellán.

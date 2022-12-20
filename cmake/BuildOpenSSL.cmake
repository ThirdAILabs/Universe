# MIT License
#
# Copyright (c) 2015-2021 The ViaDuck Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# Build OpenSSL locally.

# 2022 December 16: Adapted for cross-compilation use for the M1 wheel by
# @jerin-thirdai, trimming some excess. Original source is from
# https://github.com/viaduck/openssl-cmake/, from commit
# 777190bc4bdda3d48de3b5dd9c804669942b2210

# includes
include(ProcessorCount)
include(ExternalProject)

# Parallelize OpenSSL build as an external project.
ProcessorCount(NUM_JOBS)

if(OPENSSL_BUILD_HASH)
  set(OPENSSL_CHECK_HASH URL_HASH SHA256=${OPENSSL_BUILD_HASH})
endif()

# if already built, do not build again
if((EXISTS ${OPENSSL_LIBSSL_PATH}) AND (EXISTS ${OPENSSL_LIBCRYPTO_PATH}))
  message(
    WARNING
      "Not building OpenSSL again. Remove ${OPENSSL_LIBSSL_PATH} and ${OPENSSL_LIBCRYPTO_PATH} for rebuild"
  )
else()
  if(NOT OPENSSL_BUILD_VERSION)
    message(FATAL_ERROR "You must specify OPENSSL_BUILD_VERSION!")
  endif()

  # for OpenSSL we can only use GNU make, no exotic things like Ninja (MSYS
  # always uses GNU make)
  find_program(MAKE_PROGRAM make)

  # save old git values for core.autocrlf and core.eol
  execute_process(
    COMMAND ${GIT_EXECUTABLE} config --global --get core.autocrlf
    OUTPUT_VARIABLE GIT_CORE_AUTOCRLF
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} config --global --get core.eol
    OUTPUT_VARIABLE GIT_CORE_EOL
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(PERL_PATH_FIX_INSTALL true)

  # user-specified modules
  set(CONFIGURE_OPENSSL_MODULES ${OPENSSL_MODULES})

  # additional configure script parameters
  set(CONFIGURE_OPENSSL_PARAMS --libdir=lib)
  if(OPENSSL_DEBUG_BUILD)
    set(CONFIGURE_OPENSSL_PARAMS
        "${CONFIGURE_OPENSSL_PARAMS} no-asm -g3 -O0 -fno-omit-frame-pointer -fno-inline-functions"
    )
  endif()

  # set install command depending of choice on man page generation
  if(OPENSSL_INSTALL_MAN)
    set(INSTALL_OPENSSL_MAN "install_docs")
  endif()

  # disable building tests
  if(NOT OPENSSL_ENABLE_TESTS)
    set(CONFIGURE_OPENSSL_MODULES ${CONFIGURE_OPENSSL_MODULES} no-tests)
    set(COMMAND_TEST "true")
  endif()

  # cross-compiling @jerin-thirdai: This is added to convert the string into a
  # cmake-list.
  separate_arguments(CONFIGURE_OPENSSL_MODULES)

  if(OPENSSL_CROSS_COMPILE_MACOSX_ARM)
    set(OPENSSL_CROSS_COMPILE_TARGET darwin64-arm64-cc)
    message(STATUS "(cross)-Compiling for MacOSX ARM")
    set(COMMAND_CONFIGURE
        ./Configure ${CONFIGURE_OPENSSL_PARAMS} ${OPENSSL_CROSS_COMPILE_TARGET}
        ${CONFIGURE_OPENSSL_MODULES} --prefix=/usr/local/)
    set(COMMAND_TEST "true")

  else() # detect host system automatically
    set(COMMAND_CONFIGURE ./config ${CONFIGURE_OPENSSL_PARAMS}
                          ${CONFIGURE_OPENSSL_MODULES})
    if(NOT COMMAND_TEST)
      set(COMMAND_TEST ${MAKE_PROGRAM} test)
    endif()
  endif()

  # add openssl target
  message(
    STATUS
      "Configuring ExternalProject OpenSSL for version ${OPENSSL_BUILD_VERSION}"
  )
  ExternalProject_Add(
    openssl
    URL https://www.openssl.org/source/openssl-${OPENSSL_BUILD_VERSION}.tar.gz
        ${OPENSSL_CHECK_HASH}
    UPDATE_COMMAND "" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_COMMAND ${COMMAND_CONFIGURE}
    BUILD_COMMAND ${MAKE_PROGRAM} -j ${NUM_JOBS}
    BUILD_BYPRODUCTS ${OPENSSL_LIBSSL_PATH} ${OPENSSL_LIBCRYPTO_PATH}
    TEST_BEFORE_INSTALL 1
    TEST_COMMAND ${COMMAND_TEST}
    INSTALL_COMMAND ${MAKE_PROGRAM} DESTDIR=${CMAKE_CURRENT_BINARY_DIR}
                    install_sw ${INSTALL_OPENSSL_MAN}
    COMMAND ${CMAKE_COMMAND} -G ${CMAKE_GENERATOR}
            ${CMAKE_BINARY_DIR} # force CMake-reload
    LOG_INSTALL 1
    DOWNLOAD_NO_PROGRESS 1
    BUILD_IN_SOURCE 1)
endif()

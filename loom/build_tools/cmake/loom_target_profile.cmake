# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Typed compiler identities projected from the Bazel profile declarations.
function(loom_target_profile)
  cmake_parse_arguments(_RULE "" "NAME;FAMILY;SELECTOR" "REQUIRES" ${ARGN})
  set(_AVAILABLE TRUE)
  if(DEFINED _RULE_REQUIRES)
    if(NOT (${_RULE_REQUIRES}))
      set(_AVAILABLE FALSE)
    endif()
  endif()
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_library("${_TARGET}" INTERFACE)
  add_library("${_PACKAGE_NS}::${_RULE_NAME}" ALIAS "${_TARGET}")
  set_target_properties("${_TARGET}" PROPERTIES
    LOOM_COMPILER_TARGET "${_RULE_FAMILY}:${_RULE_SELECTOR}"
    LOOM_PROFILE_NAME "${_RULE_NAME}"
    LOOM_PROFILE_AVAILABLE "${_AVAILABLE}"
    LOOM_TARGET_PROFILES "${_TARGET}")
endfunction()

# Ordered target collections projected from Bazel target-set declarations.
function(loom_target_set)
  cmake_parse_arguments(_RULE "ALLOW_EMPTY" "NAME" "TARGETS" ${ARGN})
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_PROFILES)
  foreach(_MEMBER ${_RULE_TARGETS})
    iree_package_target_name(_MEMBER_TARGET "${_MEMBER}")
    if(NOT TARGET "${_MEMBER_TARGET}")
      message(SEND_ERROR
        "loom_target_set(${_RULE_NAME}): unknown target ${_MEMBER}")
      continue()
    endif()
    get_property(_IS_TARGET_SET
      TARGET "${_MEMBER_TARGET}" PROPERTY LOOM_TARGET_PROFILES SET)
    if(NOT _IS_TARGET_SET)
      message(SEND_ERROR
        "loom_target_set(${_RULE_NAME}): ${_MEMBER} is not a Loom target profile or set")
      continue()
    endif()
    get_target_property(_MEMBER_PROFILES
      "${_MEMBER_TARGET}" LOOM_TARGET_PROFILES)
    if(_MEMBER_PROFILES)
      list(APPEND _PROFILES ${_MEMBER_PROFILES})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _PROFILES)
  if(NOT _PROFILES AND NOT _RULE_ALLOW_EMPTY)
    message(SEND_ERROR
      "loom_target_set(${_RULE_NAME}) must contain at least one target profile")
  endif()
  add_library("${_TARGET}" INTERFACE)
  add_library("${_PACKAGE_NS}::${_RULE_NAME}" ALIAS "${_TARGET}")
  set_target_properties("${_TARGET}" PROPERTIES
    LOOM_TARGET_PROFILES "${_PROFILES}")
endfunction()

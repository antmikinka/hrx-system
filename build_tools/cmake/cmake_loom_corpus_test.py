# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import bazel_to_cmake_config
import bazel_to_cmake_requirements
import bazel_to_cmake_targets

from build_tools.cmake.test_environment import (
    REPO_ROOT,
    configure_project,
    test_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/loom_corpus"


class CMakeLoomCorpusTest(unittest.TestCase):
    def test_profile_requirements_control_registered_execution_tests(self):
        project = bazel_to_cmake_config.include_project(
            str(REPO_ROOT / ".bazel_to_cmake.cfg.py"),
            "loom/.bazel_to_cmake.cfg.py",
        )
        with tempfile.TemporaryDirectory(prefix="cmake loom corpus ") as temporary:
            source, build = Path(temporary) / "source", Path(temporary) / "build"
            shutil.copytree(FIXTURE, source)
            converter = SimpleNamespace(body="")
            functions = project.build_file_functions(
                converter=converter,
                targets=bazel_to_cmake_targets.TargetConverter(repo_map={"@hrx": ""}),
                build_dir=str(source),
                repo_root=str(REPO_ROOT),
            )
            manifest = functions.loom_corpus_manifest(
                "examples", "//fixture", ["identity.loom"]
            )
            functions.loom_corpus_sources("sources", manifest)
            profiles = []
            for name, expression in (
                ("unconditional", None),
                ("single", "FEATURE_ENABLED"),
                (
                    "compound",
                    "FEATURE_ENABLED AND (FEATURE_DISABLED OR FEATURE_ENABLED)",
                ),
                ("disabled", "FEATURE_DISABLED"),
                (
                    "compound_disabled",
                    "FEATURE_ENABLED AND (FEATURE_DISABLED OR FEATURE_MISSING)",
                ),
                ("literal_false", "FALSE"),
            ):
                requirements = []
                if expression is not None:
                    requirements.append(
                        bazel_to_cmake_requirements.build_requirement(
                            name, "//config:" + name, name, expression
                        )
                    )
                functions.loom_target_profile(
                    name,
                    "amdgpu",
                    "gfx90a",
                    target_compatible_with=[]
                    if expression is None
                    else [
                        bazel_to_cmake_requirements.CMakeRequirementCondition(
                            expression
                        )
                    ],
                )
                profiles.append(
                    functions.loom_execution_profile(
                        name,
                        "amdgpu",
                        "hardware",
                        "amdgpu",
                        build_requirements=requirements,
                    )
                )
            functions.loom_corpus_test(
                "execution",
                functions.loom_corpus_catalog([manifest]),
                execution_profiles=profiles,
            )
            converted = source / "converted.cmake"
            converted.write_text(converter.body)
            for enabled in (True, False):
                with self.subTest(enabled=enabled):
                    configure_project(
                        source,
                        build,
                        f"-DLOOM_CONVERTED_RULES={converted}",
                        f"-DFEATURE_ENABLED={'ON' if enabled else 'OFF'}",
                        "-DFEATURE_DISABLED=OFF",
                    )
                    tests = json.loads(test_project(build, "--show-only=json-v1"))[
                        "tests"
                    ]
                    expected_profiles = {"unconditional"}
                    if enabled:
                        expected_profiles.update(("single", "compound"))
                    self.assertEqual(
                        set(
                            (build / "available_profiles.txt").read_text().splitlines()
                        ),
                        expected_profiles,
                    )
                    self.assertEqual(
                        {test["name"] for test in tests},
                        {
                            f"fixture/examples_identity_test_execute_{name}_test"
                            for name in expected_profiles
                        },
                    )


if __name__ == "__main__":
    unittest.main()

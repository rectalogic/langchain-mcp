# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import pytest
from langchain_tests.unit_tests import ToolsUnitTests


@pytest.mark.usefixtures("mcptool")
class TestMCPToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return self.tool

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"path": "LICENSE"}

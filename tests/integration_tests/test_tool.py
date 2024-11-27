# Copyright (C) 2024 Andrew Wason
# SPDX-License-Identifier: MIT

import pytest
from langchain_tests.integration_tests import ToolsIntegrationTests


@pytest.mark.usefixtures("mcptool")
class TestMCPToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self):
        return self.tool

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"path": "LICENSE"}

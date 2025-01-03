import importlib.util
import json
import sys
from io import StringIO
from types import ModuleType
from typing import Annotated, Any, Dict, List, Optional, Type, TypeVar, Union

from datamodel_code_generator import DataModelType, InputFileType, generate
from pydantic import BaseModel, EmailStr, Field


def schema_to_pydantic(schema: Union[str, Dict], model_name: str = "Model") -> Type[Any]:
    """
    Convert JSON Schema to Pydantic model class.

    Args:
        schema: JSON schema as string or dictionary
        model_name: Name for the generated model class

    Returns:
        Type[Any]: Generated Pydantic model class
    """
    # Convert dict to JSON string if necessary
    if isinstance(schema, dict):
        schema = json.dumps(schema)

    # Set up a string buffer to capture stdout
    old_stdout = sys.stdout
    string_buffer = StringIO()
    sys.stdout = string_buffer

    try:
        # Generate the model code
        generate(
            input_=schema,
            input_file_type=InputFileType.JsonSchema,
            output_model_type=DataModelType.PydanticBaseModel,
            output=None,
            class_name=model_name,
            use_schema_description=True,
            field_constraints=True,
            snake_case_field=True,
            strip_default_none=True,
            disable_timestamp=True,
        )

        # Get the generated code and clean it up
        generated_code = string_buffer.getvalue()

        # Create a new module for our model
        module_name = f"dynamic_model_{model_name}"
        module = ModuleType(module_name)

        # Add all necessary types to the module's namespace
        module.Optional = Optional
        module.List = List
        module.Dict = Dict
        module.Any = Any
        module.Union = Union
        module.TypeVar = TypeVar
        module.Annotated = Annotated
        module.Type = Type
        module.BaseModel = BaseModel
        module.Field = Field
        module.EmailStr = EmailStr

        # Add the module to sys.modules so type evaluation can find it
        sys.modules[module_name] = module

        # Execute the code in the module's namespace
        exec(generated_code, module.__dict__)

        # Get the model class
        model_class = getattr(module, model_name)

        # Transfer the module's namespace to the model class
        model_class.__module__ = module_name

        return model_class

    finally:
        sys.stdout = old_stdout
        string_buffer.close()


# Example usage
if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "SELECT SQL query to execute"}},
        "required": ["query"],
    }

    # Generate Pydantic model class
    read_query_schema = schema_to_pydantic(schema, "read_query_schema_blah")

    # Test the model
    try:
        # Create a person with all fields
        person1 = read_query_schema(query="John Doe")
        print("Person 1:", person1.model_dump())
    except Exception as e:
        print(f"Error creating person: {str(e)}")

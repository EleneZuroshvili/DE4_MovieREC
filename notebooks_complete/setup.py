import dotenv

dotenv.load_dotenv()

from agents import FunctionTool


def bedrock_tool(tool: dict) -> FunctionTool:
    """
    Converts an OpenAI tool to a Bedrock tool.
    """

    return FunctionTool(
        name=tool["name"],
        description=tool["description"],
        params_json_schema={
            "type": "object",
            "properties": {
                k: v for k, v in tool["params_json_schema"]["properties"].items()
            },
            "required": tool["params_json_schema"].get("required", []),
        },
        on_invoke_tool=tool["on_invoke_tool"],
    )

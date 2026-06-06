"""
image_gen_server.py

MCP server exposing image generation via OpenAI-compatible APIs.
"""

import asyncio
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import GENERATED_IMAGES_DIR

server = Server("image_gen")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_image",
            description="Generate an image using an image-capable model (e.g. gpt-image-1)",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image description prompt"},
                    "model": {"type": "string", "description": "Model name (auto-detects if omitted)"},
                    "size": {"type": "string", "description": "Image size (default 1024x1024)"},
                    "quality": {"type": "string", "description": "Quality: low, medium, high, auto (default medium)"},
                },
                "required": ["prompt"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "generate_image":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    prompt = arguments.get("prompt", "")
    model_spec = arguments.get("model", "")
    size = arguments.get("size", "1024x1024")
    quality = arguments.get("quality", "medium")

    if not prompt:
        return [TextContent(type="text", text="Error: Image prompt is required")]

    try:
        from src.settings import get_setting
        from src.ai_interaction import do_generate_image

        if not get_setting("image_gen_enabled", True):
            return [TextContent(type="text", text="Error: Image generation is disabled by the administrator.")]

        # Delegate to the single shared implementation in ai_interaction, which
        # handles model auto-detect, OpenAI /images/generations AND the OpenRouter
        # chat-completions-with-modalities path, plus gallery persistence. The
        # content format is newline-delimited: prompt, model, size, quality.
        content = "\n".join([prompt, model_spec, size, quality])
        res = await do_generate_image(content)

        if not isinstance(res, dict) or res.get("error"):
            err = (res or {}).get("error", "unknown error") if isinstance(res, dict) else "unknown error"
            return [TextContent(type="text", text=f"Error: {err}")]

        result = (
            f"Generated image for: {str(res.get('image_prompt', prompt))[:100]}\n"
            f"image_url: {res.get('image_url')}\n"
            f"model: {res.get('image_model')}\n"
            f"size: {res.get('image_size')}"
        )
        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())

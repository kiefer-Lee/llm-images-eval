"""Quick smoke test: send one image to the VLM and print the raw response."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vlm_client import OpenAICompatibleVlmClient, VlmConfig
from src.image_ops import prepare_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one image to a VLM and print the raw response.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--prompt", default="Describe what you see in this image.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--image-max-side", type=int, default=1280)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full raw API response (choices, content, reasoning_content) for debugging.",
    )
    args = parser.parse_args()

    prepared = prepare_image(args.image, max_side=args.image_max_side)
    print(f"Image: {args.image}")
    print(f"Sent size: {prepared.transform.sent_width}x{prepared.transform.sent_height}")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url or '(default OpenAI)'}")
    print("-" * 60)

    config = VlmConfig(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
    )
    client = OpenAICompatibleVlmClient(config)

    if args.raw:
        _print_raw(client, args.prompt, prepared.data_url)
        return

    response = client.detect(args.prompt, prepared.data_url)
    print(response)


def _print_raw(client: "OpenAICompatibleVlmClient", prompt: str, image_data_url: str) -> None:
    """Call the API directly and dump every field so we can see where output lands."""
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    create_kwargs = {
        "model": client.config.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": client.config.temperature,
    }
    if client.config.max_tokens is not None:
        create_kwargs["max_tokens"] = client.config.max_tokens
    response = client.client.chat.completions.create(**create_kwargs)
    if isinstance(response, str):
        print("[raw response is a plain string]")
        print(response)
        return

    print("type:", type(response).__name__)
    try:
        print(response.model_dump_json(indent=2))
    except Exception:  # noqa: BLE001 - fall back to per-field inspection.
        message = response.choices[0].message
        print("content:", repr(getattr(message, "content", None)))
        print("reasoning_content:", repr(getattr(message, "reasoning_content", None)))
        print("finish_reason:", repr(response.choices[0].finish_reason))


if __name__ == "__main__":
    main()

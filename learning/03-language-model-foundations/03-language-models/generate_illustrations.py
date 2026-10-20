"""Generate curriculum illustrations with MAI-Image-2.6 using your Azure CLI sign-in.

Run from any directory: python generate_illustrations.py [--manifest PROMPTS.json] [--only FILENAME]
The default manifest is this chapter's 03A prompt file. Images are written beside the manifest.
Existing images are kept unless --force is supplied. Requires Azure CLI sign-in.
Adapted from the helper used for the original Chapter 03 curriculum images.
API: https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-mai-image
"""

import argparse
import base64
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENDPOINT = (
    "https://070-deepaksha-testfound-resource.services.ai.azure.com"
    "/mai/v1/images/generations?api-version=preview"
)
MODEL = "MAI-Image-2.6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=HERE / "images" / "03a-illustration-prompts.json",
        help="Prompt manifest; images are saved in the same folder.",
    )
    parser.add_argument("--only", help="Generate one filename from the prompt manifest.")
    parser.add_argument("--force", action="store_true", help="Replace existing images.")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    image_dir = manifest.parent
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if args.only:
        entries = [entry for entry in entries if entry["filename"] == args.only]
        if not entries:
            parser.error(f"Unknown illustration: {args.only}")
    pending = [
        entry for entry in entries
        if args.force or not (image_dir / entry["filename"]).exists()
    ]
    if not pending:
        print("All requested images already exist. Use --force to regenerate.")
        return
    azure_cli = shutil.which("az")
    if azure_cli is None:
        raise RuntimeError("Azure CLI is missing. Install it and run az login first.")
    token = subprocess.run(
        [
            azure_cli, "account", "get-access-token",
            "--resource", "https://cognitiveservices.azure.com",
            "--query", "accessToken", "-o", "tsv",
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not token:
        raise RuntimeError("Azure CLI returned no access token. Run az login again.")
    for entry in pending:
        prompt = (
            entry["prompt"]
            + " Required elements: " + entry["acceptance_criteria"]
            + " Do not include: " + entry["prohibited_content"]
        )
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps({
                "model": MODEL, "prompt": prompt,
                "width": 1536, "height": 864, "n": 1,
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        print(f"Generating {entry['filename']} ...", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1500]
            raise RuntimeError(f"Foundry returned HTTP {error.code}: {detail}") from error
        raw = base64.b64decode(result["data"][0]["b64_json"], validate=True)
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Foundry did not return a PNG; no output was written.")
        output = image_dir / entry["filename"]
        output.write_bytes(raw)
        print(f"Saved {output.name} ({len(raw):,} bytes)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Remove unnecessary emojis from markdown files.

Usage:
    python remove-emojis.py [directory]
    
If no directory specified, processes all files under notes/

Rules:
1. Remove decorative emojis (🎯, 🚀, ✅, ⚡, 📊, 💡, ⚠️, etc.)
2. Replace emoji-based callouts with text equivalents:
   - "💡 **Checkpoint:**" → "**Checkpoint:**"
   - "⚠️ **Warning:**" → "**Warning:**"
   - "🎯 **Rule of Thumb:**" → "**Rule of Thumb:**"
   - etc.
"""

import re
import os
from pathlib import Path

# Define emoji patterns and their text replacements
EMOJI_REPLACEMENTS = [
    # Callout patterns - specific replacements
    (r'🎯\s*\*\*Rule of Thumb:\*\*', '**Rule of Thumb:**'),
    (r'💡\s*\*\*([^*]+):\*\*', r'**\1:**'),  # Generic insight pattern
    (r'⚠️\s*\*\*([^*]+):\*\*', r'**Warning — \1:**'),  # Warning pattern
    (r'>\s*💡\s*\*\*([^*]+):\*\*', r'> **\1:**'),  # Blockquote insight
    (r'>\s*⚠️\s*\*\*([^*]+):\*\*', r'> **\1:**'),  # Blockquote warning
    (r'>\s*\*\*⚠️\s*([^*]+):\*\*', r'> **\1:**'),  # Blockquote warning variant

    # Table and summary headers
    (r'<strong>📊\s*([^<]+)</strong>', r'<strong>\1</strong>'),
    (r'<strong>📚\s*([^<]+)</strong>', r'<strong>\1</strong>'),
    (r'<strong>✏️\s*([^<]+)</strong>', r'<strong>\1</strong>'),

    # Standalone emojis at start of lines
    (r'^\s*🎯\s+', '', re.MULTILINE),
    (r'^\s*💡\s+', '', re.MULTILINE),
    (r'^\s*⚠️\s+', '', re.MULTILINE),
    (r'^\s*✅\s+', '', re.MULTILINE),
    (r'^\s*❌\s+', '', re.MULTILINE),
    (r'^\s*⚡\s+', '', re.MULTILINE),
    (r'^\s*📊\s+', '', re.MULTILINE),
    (r'^\s*📚\s+', '', re.MULTILINE),
    (r'^\s*✏️\s+', '', re.MULTILINE),
    (r'^\s*🌟\s+', '', re.MULTILINE),
    (r'^\s*🏗️\s+', '', re.MULTILINE),
    (r'^\s*🍽️\s+', '', re.MULTILINE),
    (r'^\s*🛤️\s+', '', re.MULTILINE),
    (r'^\s*⏸️\s+', '', re.MULTILINE),
    (r'^\s*⏭️\s+', '', re.MULTILINE),
    (r'^\s*➡️\s+', '', re.MULTILINE),
]

# Common emoji characters to remove (comprehensive list)
EMOJI_CHARS = '🎯🚀✅⚡📊💡⚠️🔍📝🌟⭐🎨🔧🛠️💪🏆🎓📚🔥💻🖥️📈📉🎭🎬🎮🎪🎉🎊🎈🔔🔕🔒🔓🔑🗝️🏃‍♂️🏃‍♀️🏃🚶‍♂️🚶‍♀️🚶🧠🤖🤔💭💬🗨️🗯️💥⚙️🔬🧪🧬🔭🗂️📁📂🗃️📋📌📍📎🖇️📏📐✂️🖊️🖍️✏️🖋️✒️📄📃📑🗒️🗓️📅📆🗄️🗑️🏗️🍽️🛤️⏸️⏭️❌'

def remove_emojis_from_file(filepath):
    """Remove emojis from a single markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        emoji_count = 0

        # Apply specific replacements
        for pattern, replacement, *flags in EMOJI_REPLACEMENTS:
            flag = flags[0] if flags else 0
            new_content = re.sub(pattern, replacement, content, flags=flag)
            emoji_count += len(re.findall(pattern, content, flags=flag))
            content = new_content

        # Remove remaining standalone emojis (not in patterns above)
        # Count them first
        for char in EMOJI_CHARS:
            emoji_count += content.count(char)

        # Remove them
        for char in EMOJI_CHARS:
            content = content.replace(char, '')

        # Clean up any double spaces or trailing spaces that may have been created
        content = re.sub(r'  +', ' ', content)
        content = re.sub(r' +\n', '\n', content)

        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return emoji_count
        return 0

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0

def process_directory(directory):
    """Process all markdown files in directory and subdirectories."""
    directory = Path(directory)
    results = {}

    for md_file in directory.rglob('*.md'):
        emoji_count = remove_emojis_from_file(md_file)
        if emoji_count > 0:
            results[str(md_file)] = emoji_count

    return results

def main():
    import sys
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path(__file__).parent.parent / 'notes'

    print(f"Processing markdown files in: {target_dir}")
    print("=" * 60)

    results = process_directory(target_dir)

    print("\nResults:")
    print("=" * 60)
    total_emojis = 0
    for filepath, count in sorted(results.items()):
        try:
            rel_path = Path(filepath).relative_to(target_dir)
        except ValueError:
            rel_path = Path(filepath)
        print(f"{rel_path}: {count} emojis removed")
        total_emojis += count

    print("=" * 60)
    print(f"Total files processed: {len(results)}")
    print(f"Total emojis removed: {total_emojis}")

if __name__ == '__main__':
    main()

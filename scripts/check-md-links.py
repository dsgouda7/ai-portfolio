import re
from pathlib import Path
repo = Path('.').resolve()
pattern = re.compile(r'\[[^\]]*\]\(([^)]+)\)')
mds = [p for p in repo.rglob('*.md') if '.venv' not in p.parts]
gen_refs = []
broken = []
for md in mds:
    try:
        text = md.read_text(encoding='utf-8')
    except Exception:
        continue
    for m in pattern.findall(text):
        link = m.strip()
        if link.startswith('http'):
            continue
        if link.startswith('#'):
            continue
        link_no_anchor = link.split('#',1)[0]
        if 'GenScripts' in link_no_anchor:
            gen_refs.append(f"{md} -> {link}")
        target = Path(link_no_anchor)
        if not target.is_absolute():
            try:
                target = (md.parent / link_no_anchor).resolve()
            except Exception:
                # Fallback: skip resolve on Windows path-too-long issues
                target = (md.parent / link_no_anchor)
        try:
            exists = target.exists()
        except Exception:
            exists = False
        if not exists:
            broken.append(f"{md} -> {link}")
if gen_refs:
    print('FOUND_GENSCRIPTS_REFS:')
    for g in sorted(set(gen_refs)):
        print(g)
else:
    print('NO_GENSCRIPTS_REFS_FOUND')
if broken:
    print('\nBROKEN_LINKS:')
    for b in sorted(set(broken)):
        print(b)
else:
    print('\nNO_BROKEN_LOCAL_LINKS')

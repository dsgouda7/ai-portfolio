import os
import json
import re

notebook_dir = r"c:\r\ai-portfolio\learning\genai\01-transformers"
notebook_files = sorted([f for f in os.listdir(notebook_dir) if f.endswith('.ipynb')])

out_lines = []

for filename in notebook_files:
    filepath = os.path.join(notebook_dir, filename)
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    cell_count = len(cells)
    
    out_lines.append("="*60)
    out_lines.append(f"FILE: {filename}")
    out_lines.append(f"Cell Count: {cell_count}")
    
    # Extract headings (1-based index)
    headings = []
    first_md_cell_src = None
    math_heavy_cells = []
    
    for idx, cell in enumerate(cells, 1):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        if isinstance(source, list):
            source_str = "".join(source)
        else:
            source_str = str(source)
            
        if cell_type == 'markdown':
            if first_md_cell_src is None:
                first_md_cell_src = source_str
                
            # Headings
            for line in source_str.split('\n'):
                line_stripped = line.strip()
                if line_stripped.startswith('# '):
                    headings.append(f"Cell {idx} (H1): {line_stripped[2:]}")
                elif line_stripped.startswith('## '):
                    headings.append(f"Cell {idx} (H2): {line_stripped[3:]}")
            
            # Identify math-heavy cells: search for $$ or multiple $ expressions
            # e.g., if there's substantial LaTeX formulas before intuitive code/text
            dollar_count = source_str.count('$')
            if dollar_count > 6 or '$$' in source_str:
                math_heavy_cells.append((idx, source_str[:120].replace('\n', ' ') + "..."))
                
    out_lines.append("\nHEADINGS:")
    for h in headings:
        out_lines.append(f"  {h}")
        
    out_lines.append("\nFIRST MARKDOWN CELL SOURCE:")
    out_lines.append("-" * 40)
    out_lines.append(first_md_cell_src if first_md_cell_src else "None")
    out_lines.append("-" * 40)
    
    # Word Counts on combined text source (case insensitive)
    # Merging all cells source
    all_text = ""
    for cell in cells:
        source = cell.get('source', [])
        if isinstance(source, list):
            all_text += "".join(source) + "\n"
        else:
            all_text += source + "\n"
    
    # We want counts for Riverside, aria, cat, token, embedding, positional/position, QKV or Q/K/V, FFN/feed-forward, logits, softmax, loss, backprop/gradient
    def count_pattern(raw_text, pattern_regex):
        return len(re.findall(pattern_regex, raw_text, re.IGNORECASE))
        
    riverside_count = count_pattern(all_text, r'\briverside\b')
    aria_count = count_pattern(all_text, r'\baria\b')
    cat_word_count = count_pattern(all_text, r'\bcat\b')
    cat_all_count = count_pattern(all_text, r'cat') # just substring
    
    token_count = count_pattern(all_text, r'token') # substring
    embed_count = count_pattern(all_text, r'embed') # substring
    position_count = count_pattern(all_text, r'position') # substring
    
    qkv_count = count_pattern(all_text, r'qkv')
    q_k_v_count = count_pattern(all_text, r'q/k/v')
    ffn_count = count_pattern(all_text, r'ffn')
    feed_forward_count = count_pattern(all_text, r'feed[- ]forward')
    
    logits_count = count_pattern(all_text, r'logits?') # count logit or logits
    softmax_count = count_pattern(all_text, r'softmax')
    loss_count = count_pattern(all_text, r'\bloss\b') # use word boundary for loss
    backprop_count = count_pattern(all_text, r'backprop')
    gradient_count = count_pattern(all_text, r'gradient')
    
    out_lines.append("\nTERM COUNTS:")
    out_lines.append(f"  Riverside: {riverside_count}")
    out_lines.append(f"  aria: {aria_count}")
    out_lines.append(f"  cat (word): {cat_word_count} (substring 'cat': {cat_all_count})")
    out_lines.append(f"  token: {token_count}")
    out_lines.append(f"  embedding/embed: {embed_count}")
    out_lines.append(f"  positional/position: {position_count}")
    out_lines.append(f"  QKV: {qkv_count} | Q/K/V: {q_k_v_count}")
    out_lines.append(f"  FFN: {ffn_count} | feed-forward/feed forward: {feed_forward_count}")
    out_lines.append(f"  logits: {logits_count}")
    out_lines.append(f"  softmax: {softmax_count}")
    out_lines.append(f"  loss: {loss_count}")
    out_lines.append(f"  backprop: {backprop_count} | gradient: {gradient_count}")
    
    out_lines.append(f"\nPOTENTIAL MATH-HEAVY CELLS (by heuristic):")
    for idx, snippet in math_heavy_cells[:10]:
        out_lines.append(f"  Cell {idx}: {snippet}")
    out_lines.append("\n" + "="*60 + "\n")

with open('report_utf8.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(out_lines))

print("Successfully written report_utf8.txt")

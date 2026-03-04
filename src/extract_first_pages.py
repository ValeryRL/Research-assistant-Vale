import os
import sys
import subprocess

try:
    import fitz
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    import fitz

current_dir = os.path.dirname(os.path.abspath(__file__))
papers_dir = os.path.join(os.path.dirname(current_dir), "papers")
output_file = os.path.join(os.path.dirname(current_dir), "papers_first_pages.txt")

with open(output_file, "w", encoding="utf-8") as out:
    for f in sorted(os.listdir(papers_dir)):
        if f.endswith('.pdf'):
            out.write(f"=== DOC: {f} ===\n")
            try:
                doc = fitz.open(os.path.join(papers_dir, f))
                text = doc[0].get_text()
                # Take up to 2000 characters of the first page
                out.write(text[:2000] + "\n\n")
                doc.close()
            except Exception as e:
                out.write(f"ERROR reading {f}: {e}\n\n")

print(f"Extraction complete. Results saved to {output_file}")

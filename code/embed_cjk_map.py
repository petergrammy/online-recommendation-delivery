"""Embed the standard Unicode map for XeLaTeX's Fandol fonts in a copy of a PDF.

Usage: python embed_cjk_map.py compiled.pdf searchable.pdf
The map is supplied by TeX Live. Page content and original PDF objects are
retained through an incremental update.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject


def fonts_in_resources(resources, visited):
    """Include fonts in imported forms as well as fonts in the page's own text."""
    if id(resources) in visited:
        return
    visited.add(id(resources))
    if '/Font' in resources:
        for entry in resources['/Font'].values():
            yield entry.get_object()
    if '/XObject' in resources:
        for entry in resources['/XObject'].values():
            form = entry.get_object()
            if form.get('/Subtype') == '/Form' and '/Resources' in form:
                yield from fonts_in_resources(form['/Resources'], visited)


def needs_unicode_map(font):
    if ('/ToUnicode' in font or font.get('/Encoding') != '/Identity-H'
            or font.get('/Subtype') != '/Type0' or 'Fandol' not in font.get('/BaseFont', '')):
        return False
    info = font['/DescendantFonts'][0].get_object()['/CIDSystemInfo']
    return info.get('/Registry') == 'Adobe' and info.get('/Ordering') == 'GB1'


def load_unicode_map():
    lookup = subprocess.run(['kpsewhich', '-format=cmap', 'Adobe-GB1-UCS2'],
                            capture_output=True, text=True)
    if lookup.returncode or not lookup.stdout.strip():
        raise FileNotFoundError('TeX Live could not locate the Adobe-GB1-UCS2 CMap.')
    stream = DecodedStreamObject()
    stream.set_data(Path(lookup.stdout.strip()).read_bytes())
    return stream.flate_encode()


def embed_cjk_map(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or (target.exists() and source.samefile(target)):
        raise ValueError('Source and target PDFs must be different files.')

    writer = PdfWriter(source, incremental=True)
    visited = set()
    cmap = None
    count = 0
    for page in writer.pages:
        if '/Resources' not in page:
            continue
        for font in fonts_in_resources(page['/Resources'], visited):
            if needs_unicode_map(font):
                if cmap is None:
                    cmap = writer._add_object(load_unicode_map())
                font[NameObject('/ToUnicode')] = cmap
                count += 1

    # Replace the destination only after the complete copy has been written.
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        if count:
            writer.write(temporary)
        else:
            shutil.copyfile(source, temporary)
        shutil.copymode(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    print(f'Embedded a Unicode map for {embed_cjk_map(args.source, args.target)} Fandol fonts.')

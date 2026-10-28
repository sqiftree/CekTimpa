"""Render maintained project reference content from README."""
from pathlib import Path


def dataset_reference_markdown():
    readme = Path(__file__).resolve().parents[1] / 'README.md'
    text = readme.read_text(encoding='utf-8')
    start = '<!-- dataset-reference:start -->'
    end = '<!-- dataset-reference:end -->'
    if start not in text or end not in text:
        raise ValueError('README dataset-reference markers are missing')
    return text.split(start, 1)[1].split(end, 1)[0].strip()

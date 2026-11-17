"""Check English release documentation and annotations, including nested archives."""
import argparse
import ast
import gzip
import io
import json
from pathlib import Path
import re
import tokenize
import zipfile

HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]')
TEXT = {'.md', '.txt', '.sh', '.cfg', '.ini', '.toml', '.yaml', '.yml', '.lock', '.csv', ''}


def check(root):
    assert root.is_dir() and (root / 'metadata/language_policy.json').is_file(), 'Use the extracted current artifact root'
    findings = []
    retained_data = []
    inspected = 0

    def visit(name, data):
        nonlocal inspected
        inspected += 1
        if HAN.search(name):
            findings.append(dict(path=name, rule='non-English-file-name'))
        suffix = Path(name).suffix.lower()
        if suffix == '.zip':
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for member in z.infolist():
                    if not member.is_dir():
                        visit(name + '!' + member.filename, z.read(member))
        elif suffix == '.gz':
            visit(name[:-3], gzip.decompress(data))
        elif suffix == '.py':
            text = data.decode('utf-8')
            tree = ast.parse(text)
            docstrings = set()
            protected_inputs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.body and isinstance(node.body[0], ast.Expr):
                        value = node.body[0].value
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            docstrings.add(id(value))
                if Path(name).name == 'run_suite.py' and isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value == 'chinese':
                            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                                protected_inputs.add(id(value))
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type in {tokenize.COMMENT, tokenize.NAME} and HAN.search(token.string):
                    findings.append(dict(path=name, line=token.start[0], rule='non-English-annotation-or-identifier'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and HAN.search(node.value):
                    if id(node) in protected_inputs:
                        retained_data.append(dict(path=name, line=node.lineno, kind='fixed-original-input'))
                    else:
                        findings.append(dict(path=name, line=node.lineno,
                                             rule='non-English-docstring' if id(node) in docstrings else 'unclassified-non-English-string'))
        elif suffix in TEXT:
            text = data.decode('utf-8')
            if HAN.search(text):
                findings.append(dict(path=name, rule='non-English-documentation'))
        elif suffix in {'.json', '.jsonl'}:
            text = data.decode('utf-8')
            relative = name.rsplit('!', 1)[-1]
            metadata = relative.startswith('metadata/') or '/protocol/' in relative
            if metadata and suffix == '.json' and HAN.search(json.dumps(json.loads(text), ensure_ascii=False)):
                findings.append(dict(path=name, rule='non-English-metadata'))
            elif HAN.search(text):
                retained_data.append(dict(path=name, kind='original-recorded-text'))

    for path in sorted(root.rglob('*')):
        if path.is_file() and '.git' not in path.relative_to(root).parts:
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            visit(path.relative_to(root).as_posix(), path.read_bytes())
    return dict(pass_english_materials=not findings,
                inspected_files_including_nested=inspected,
                documentation_and_code_annotations_language='English',
                raw_text_policy='Preserve original scientific inputs and outputs without translation',
                retained_original_text_locations=retained_data,
                findings=findings, model_or_api_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = check(args.root.resolve())
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['pass_english_materials']:
        raise SystemExit(1)

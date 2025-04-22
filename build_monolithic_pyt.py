import os
import re
import sys
import ast
from pathlib import Path

# List of allowed imports (standard library + arcpy)
ALLOWED_IMPORTS = set([
    'arcpy', 'os', 'sys', 'json', 're', 'math', 'datetime', 'time', 'random', 'collections', 'itertools', 'functools', 'pathlib', 'shutil', 'logging', 'csv', 'copy', 'ast', 'typing', 'traceback', 'subprocess', 'threading', 'concurrent', 'uuid', 'base64', 'hashlib', 'tempfile', 'glob', 'inspect', 'enum', 'warnings', 'contextlib', 'io', 'zipfile', 'struct', 'platform', 'getpass', 'socket', 'http', 'urllib', 'email', 'pprint', 'argparse', 'dataclasses', 'statistics', 'string', 'types', 'site', 'importlib', 'pkgutil', 'codecs', 'signal', 'weakref', 'array', 'bisect', 'heapq', 'queue', 'resource', 'selectors', 'ssl', 'tarfile', 'xml', 'xml.etree', 'xml.dom', 'xml.sax', 'xml.parsers', 'xmlrpc', 'bz2', 'lzma', 'gzip', 'pickle', 'marshal', 'shelve', 'sqlite3', 'ctypes', 'cProfile', 'pstats', 'doctest', 'unittest', 'venv', 'ensurepip', 'distutils', 'site', 'venv', 'wsgiref', 'uuid', 'zoneinfo', 'faulthandler', 'trace', 'token', 'tokenize', 'symtable', 'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis', 'formatter', 'gettext', 'locale', 'mailbox', 'mailcap', 'mimetypes', 'mmap', 'msilib', 'netrc', 'nntplib', 'numbers', 'optparse', 'parser', 'pipes', 'poplib', 'profile', 'pydoc', 'quopri', 'reprlib', 'runpy', 'sched', 'secrets', 'selectors', 'smtpd', 'smtplib', 'sndhdr', 'spwd', 'stat', 'sunau', 'symbol', 'symtable', 'sysconfig', 'tabnanny', 'telnetlib', 'termios', 'test', 'textwrap', 'this', 'tkinter', 'turtle', 'tty', 'turtle', 'unittest', 'uu', 'venv', 'webbrowser', 'xdrlib', 'zipapp', 'zlib', 'zoneinfo'
])


def extract_imports(source_code):
    """Return a set of imported module names from the given source code."""
    tree = ast.parse(source_code)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports


def inline_code(files):
    """Concatenate code from a list of files, skipping __main__ blocks and duplicate imports."""
    seen_imports = set()
    code_blocks = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            code = f.read()
        # Remove if __name__ == '__main__' blocks
        code = re.sub(r"if __name__ ?== ?['\"]__main__['\"]:.*", '', code, flags=re.DOTALL)
        # Remove module docstrings
        code = re.sub(r'^\s*""".*?"""', '', code, flags=re.DOTALL)
        # Remove duplicate imports
        lines = code.splitlines()
        filtered_lines = []
        for line in lines:
            if line.strip().startswith('import') or line.strip().startswith('from'):
                mod = line.split()[1].split('.')[0]
                if mod in seen_imports:
                    continue
                seen_imports.add(mod)
            filtered_lines.append(line)
        code_blocks.append('\n'.join(filtered_lines))
    return '\n\n'.join(code_blocks)


def check_imports(files):
    """Check for unsupported imports in the given files."""
    unsupported = set()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            code = f.read()
        imports = extract_imports(code)
        for imp in imports:
            if imp not in ALLOWED_IMPORTS:
                unsupported.add(imp)
    return unsupported


def main():
    """Build a monolithic .pyt file from toolbox and utility modules."""
    root = Path(__file__).parent
    toolbox_file = root / 'arcgispro_ai' / 'toolboxes' / 'arcgispro_ai_tools.pyt'
    util_dir = root / 'arcgispro_ai' / 'toolboxes' / 'arcgispro_ai'  # Utility code
    util_files = [util_dir / 'arcgispro_ai_utils.py']  # Add more as needed

    # Check for unsupported imports
    unsupported = check_imports([toolbox_file] + util_files)
    if unsupported:
        print(f"WARNING: Unsupported imports found: {unsupported}")

    # Inline utility code
    util_code = inline_code(util_files)
    # Inline toolbox code
    with open(toolbox_file, 'r', encoding='utf-8') as f:
        toolbox_code = f.read()

    # Read template
    template_file = root / 'arcgispro_ai_0.0.6.pyt'
    with open(template_file, 'r', encoding='utf-8') as f:
        template = f.read()

    # Replace placeholders
    result = template.replace(
        '# --- BEGIN INLINED UTILITY CODE ---\n# (Utility functions/classes from arcgispro_ai_utils.py, etc. will be inserted here by the build script)\n# --- END INLINED UTILITY CODE ---',
        f'# --- BEGIN INLINED UTILITY CODE ---\n{util_code}\n# --- END INLINED UTILITY CODE ---'
    ).replace(
        '# --- BEGIN TOOLBOX AND TOOL CLASSES ---\n# (Toolbox and tool classes from arcgispro_ai_tools.pyt will be inserted here by the build script)\n# --- END TOOLBOX AND TOOL CLASSES ---',
        f'# --- BEGIN TOOLBOX AND TOOL CLASSES ---\n{toolbox_code}\n# --- END TOOLBOX AND TOOL CLASSES ---'
    )

    # Write output
    out_file = root / f'arcgispro_ai_0.0.6.pyt'
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"Monolithic .pyt written to {out_file}")

if __name__ == '__main__':
    main()

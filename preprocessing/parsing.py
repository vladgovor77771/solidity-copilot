import hashlib
import re

def collect_source(lines, i):
    name = lines[i].split()[1]
    source = ""
    while i < len(lines):
        source += lines[i] + '\n'
        if lines[i] == '}' or lines[i] == '{}':
            break
        i += 1
    md5_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
    return name, source, md5_hash, i

def process_source(source_code):
    solidity_version = None
    contracts = []
    libraries = []
    interfaces = []

    lines = source_code.splitlines()

    for i in range(len(lines)):
        if lines[i].startswith("pragma solidity "):
            solidity_version = lines[i].split()[2].rstrip(";")
        elif lines[i].startswith("contract "):
            name, source, md5_hash, i = collect_source(lines, i)
            contracts.append({"name": name, "source_code": source, "md5_hash": md5_hash})
        elif lines[i].startswith("library "):
            name, source, md5_hash, i = collect_source(lines, i)
            libraries.append({"name": name, "source_code": source, "md5_hash": md5_hash})
        elif lines[i].startswith("interface "):
            name, source, md5_hash, i = collect_source(lines, i)
            interfaces.append({"name": name, "source_code": source, "md5_hash": md5_hash})

    return {
        'solidity_version': solidity_version,
        'contracts': contracts,
        'libraries': libraries,
        'interfaces': interfaces,
    }


def format_declaration(decl_lines):
    return ' '.join(''.join(decl_lines).split()).replace('( ', '(').replace(' )', ')')

def format_body(body_lines):
    return '\n'.join(list(map(lambda x: x[8:], body_lines)))


def extract_functions(source: str):
    functions = []
    lines = source.splitlines()

    i = -1

    while i < len(lines) - 1:
        i += 1
        if lines[i].startswith('interface'):
            while lines[i] != '}':
                i += 1
            continue
        if lines[i].startswith('    function') or lines[i].startswith('    modifier'):
            if lines[i].endswith(';'):
                continue
            func_declaration = []
            func_body = []

            func_declaration.append(lines[i])
            bad = False
            while not lines[i].endswith('{'):
                if lines[i].endswith('{}'):
                    bad = True
                    break
                i += 1
                func_declaration.append(lines[i])
            if bad:
                continue

            i += 1
            while lines[i] != '    }':
                func_body.append(lines[i])
                i += 1

            decl = format_declaration(func_declaration)
            body = format_body(func_body)
            functions.append({
                'declaration': decl,
                'body': body,
                'hash': hashlib.md5((decl + body).encode("utf-8")).hexdigest()
            })

    return functions

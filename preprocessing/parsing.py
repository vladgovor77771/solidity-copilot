import hashlib

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
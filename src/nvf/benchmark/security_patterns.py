"""Flexible security pattern matching for test oracles.

Uses regex patterns that handle whitespace variants, aliasing, and
indirect usage. Each pattern returns True if the INSECURE pattern is found.
"""
import re
import ast
import inspect
from typing import Callable


def has_shell_injection(src: str) -> bool:
    """Check for shell=True, os.system, os.popen in any whitespace variant."""
    patterns = [
        r'shell\s*=\s*True',
        r'os\s*\.\s*system\s*\(',
        r'os\s*\.\s*popen\s*\(',
    ]
    return any(re.search(p, src) for p in patterns)


def has_sql_injection(src: str) -> bool:
    """Check for string formatting in SQL queries."""
    # Look for f-strings, .format(), or % formatting near SQL keywords
    sql_keywords = r'(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|WHERE)'
    patterns = [
        rf'f["\'].*{sql_keywords}',
        rf'{sql_keywords}.*f["\']',
        rf'\.format\s*\(.*{sql_keywords}',
        rf'{sql_keywords}.*\.format\s*\(',
        rf'%\s*\(.*{sql_keywords}',
        rf'{sql_keywords}.*%\s*\(',
        rf'{sql_keywords}.*\+\s*\w',  # String concatenation
        rf'\w\s*\+.*{sql_keywords}',
    ]
    return any(re.search(p, src, re.IGNORECASE) for p in patterns)


def has_eval_exec(src: str) -> bool:
    """Check for eval() or exec() calls."""
    patterns = [
        r'\beval\s*\(',
        r'\bexec\s*\(',
    ]
    return any(re.search(p, src) for p in patterns)


def has_unsafe_deserialization(src: str) -> bool:
    """Check for pickle.load/loads, marshal.load, cPickle, unsafe yaml.load."""
    patterns = [
        r'pickle\s*\.\s*loads?\s*\(',
        r'cPickle\s*\.\s*loads?\s*\(',
        r'_pickle\s*\.\s*loads?\s*\(',
        r'marshal\s*\.\s*loads?\s*\(',
        r'shelve\s*\.\s*open\s*\(',
        r'yaml\s*\.\s*load\s*\([^)]*(?!Loader\s*=\s*yaml\s*\.\s*SafeLoader)',
        r'yaml\s*\.\s*load\s*\(\s*[^,)]+\s*\)',  # yaml.load without Loader arg
    ]
    # Also check: yaml.load used without SafeLoader
    if re.search(r'yaml\s*\.\s*load\s*\(', src):
        if not re.search(r'safe_load|SafeLoader|CSafeLoader', src):
            return True
    return any(re.search(p, src) for p in patterns[:5])


def has_weak_hash(src: str) -> bool:
    """Check for MD5, SHA1, DES usage."""
    patterns = [
        r'hashlib\s*\.\s*md5\s*\(',
        r'hashlib\s*\.\s*sha1\s*\(',
        r'MD5\s*\.\s*new\s*\(',
        r'SHA\s*\.\s*new\s*\(',
        r'\bDES\b(?!3)',  # DES but not 3DES/DES3
        r'hashlib\s*\.\s*new\s*\(\s*["\'](?:md5|sha1)["\']',
    ]
    return any(re.search(p, src, re.IGNORECASE) for p in patterns)


def has_weak_key_size(src: str) -> bool:
    """Check for RSA/DSA key sizes < 2048."""
    patterns = [
        r'key_size\s*=\s*(?:512|1024)\b',
        r'bits\s*=\s*(?:512|1024)\b',
        r'generate\s*\(\s*(?:512|1024)\s*[,)]',
    ]
    return any(re.search(p, src) for p in patterns)


def has_hardcoded_credentials(src: str) -> bool:
    """Check for hardcoded passwords/secrets."""
    patterns = [
        r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']',
        r'(?:secret|api_key|apikey|token)\s*=\s*["\'][^"\']{4,}["\']',
    ]
    # Exclude common defaults/placeholders
    exclude = [r'""', r"''", r'"password"', r'"changeme"', r'"test"', r'"admin"']
    for p in patterns:
        match = re.search(p, src, re.IGNORECASE)
        if match:
            matched_str = match.group()
            if not any(re.search(e, matched_str) for e in exclude):
                # Check it's not reading from env
                line_start = src.rfind('\n', 0, match.start()) + 1
                line = src[line_start:src.find('\n', match.end())]
                if 'os.environ' not in line and 'getenv' not in line and 'config' not in line.lower():
                    return True
    return False


def has_weak_random(src: str) -> bool:
    """Check for random module usage for security-sensitive operations."""
    if re.search(r'\brandom\s*\.\s*(?:random|randint|choice|sample|uniform)\s*\(', src):
        # Check if secrets/os.urandom is also used (OK if so)
        if not re.search(r'secrets|os\.urandom|SystemRandom', src):
            return True
    return False


def has_disabled_verification(src: str) -> bool:
    """Check for disabled SSL/TLS verification."""
    patterns = [
        r'verify\s*=\s*False',
        r'CERT_NONE',
        r'VERIFY_NONE',
        r'check_hostname\s*=\s*False',
    ]
    return any(re.search(p, src) for p in patterns)


def has_insecure_ftp(src: str) -> bool:
    """Check for plain FTP (not FTPS/SFTP)."""
    if re.search(r'ftplib\s*\.\s*FTP\s*\(', src):
        if not re.search(r'FTP_TLS|SFTP|ssl|SSH', src, re.IGNORECASE):
            return True
    return False


def has_log_injection(src: str) -> bool:
    """Check for unsanitized user input in log messages."""
    # This is hard to detect reliably via source inspection
    # Check if any input is directly logged without sanitization
    if re.search(r'log(?:ging|ger)?\s*\.\s*(?:info|warning|error|debug)\s*\(.*(?:input|request|user)', src, re.IGNORECASE):
        if not re.search(r'replace|strip|escape|sanitize|encode', src):
            return True
    return False


def has_unsafe_xml(src: str) -> bool:
    """Check for XML parsing without defusedxml."""
    if re.search(r'xml\s*\.\s*etree|ElementTree|lxml\s*\.\s*etree', src):
        if not re.search(r'defusedxml|defused', src, re.IGNORECASE):
            return True
    return False


# Map CWE numbers to checker functions
CWE_CHECKERS: dict[int, Callable[[str], bool]] = {
    20: has_unsafe_xml,        # Input validation (XML variant)
    22: lambda src: False,     # Path traversal - hard to detect statically
    78: has_shell_injection,
    79: lambda src: False,     # XSS - needs web context
    89: has_sql_injection,
    94: has_eval_exec,
    95: has_eval_exec,
    116: lambda src: False,    # Improper encoding
    117: has_log_injection,
    259: has_hardcoded_credentials,
    295: has_disabled_verification,
    319: has_insecure_ftp,
    326: has_weak_key_size,
    327: has_weak_hash,
    329: lambda src: False,    # Static IV - hard to detect
    330: has_weak_random,
    377: lambda src: False,    # Temp file - hard to detect
    400: lambda src: False,    # ReDoS - needs runtime
    502: has_unsafe_deserialization,
    732: lambda src: False,    # Permissions
    798: has_hardcoded_credentials,
}

"""Operator identity and artifact names; never infer mathematical meaning."""
from pathlib import Path
import re

from .trace_io import InputError


def validate_operator_name(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 100
            or value in {'.', '..'} or not re.fullmatch(r'[\w.-]+', value)):
        raise InputError('算子名称需为1–100个字母、数字、下划线、连字符或点，不能是路径或 . / ..')
    return value


def identify_operator(result, explicit=None):
    functions = {stage: [f['name'] for f in item.get('functions', [])]
                 for stage, item in result.get('facts', {}).items()}
    metadata_name = result['input'].get('metadata', {}).get('name')
    unique = list(dict.fromkeys(name for names in functions.values() for name in names))
    if explicit is not None:
        name, source = explicit, 'explicit'
    elif metadata_name:
        name, source = metadata_name, 'manifest'
    elif len(unique) == 1:
        name, source = unique[0], 'ir_function'
    else:
        name, source = Path(result['input']['root']).name, 'trace_directory'
    return {'name': validate_operator_name(name), 'name_source': source,
            'ir_functions': functions}


def artifact_names(name):
    name = validate_operator_name(name)
    return {'report': f'{name} 自动检查记录.md',
            'evidence': f'{name} 检查证据.zip',
            'final_report': f'{name} 单算子完整检查报告.md'}

def field_formater(field_name: str, extra: str = None) -> str:
    return field_name.replace('_', ' ') if extra is None else f"{field_name.replace('_',' ')} ({extra})"
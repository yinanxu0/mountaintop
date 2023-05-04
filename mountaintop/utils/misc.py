from collections import Iterable
import sys
import re


def flush(string):
    """
    Flush the string to the current line and overwrite original content.

    Args:
        string: string to flush

    """
    sys.stdout.write(string.strip() + '\r')
    sys.stdout.flush()


def byteify(value, encoding='utf-8'):
    """
    Decode unicode into byte string. Do type checking.
    Guarantee to produce a byte string.

    Args:
        value: value of type unicode or str.
        encoding: unicode encoding. used to decode it into byte str

    Returns:
        value: a converted byte str

    Raises:
        TypeError

    """
    if isinstance(value, bytes):
        return value.encode(encoding)
    elif isinstance(value, str):
        return value
    else:
        raise TypeError(f"Expect type in {(bytes, str)}, got {type(value)}: {value}")


def snake2camel(snake_str, shrink_keep=0):
    """
    "a_snake_case_string" to "ASnakeCaseString"
    if shrink_keep > 0, say shrink_keep = 2
    "a_snake_case_string" to "ASnCaString"
    """
    components = snake_str.split('_')
    if shrink_keep:

        return ''.join([x[0:shrink_keep].title() if len(x) > shrink_keep else x
                        for x in components[:-1]]) + components[-1].title()
    else:
        return ''.join(x.title() for x in components)


def camel2snake(camel_str):
    """
    "ACamelCaseString" to "a_camel_case_string"
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def assure_value_type(value, dtype_or_dtype_tuple, convert_func=None, convert=True):
    """
    Assert whether a value has type specified in dtype_or_dtype_tuple.
    After assertion is passed, if convert is True:
    If dtype_or_dtype_tuple is a tuple, then value is converted according to convert_func.
    If convert_func is None, convert_func=dtype_or_dtype_tuple[0]


    Args:
        value: an object of built-in type
        dtype_or_dtype_tuple: a type object or a tuple of type object
        convert_func: type convert function when convert is True. Default to dtype_or_dtype_tuple[0]
                        if dtype_or_dtype_tuple is tuple
        convert: whether to convert according

    Returns:
        converted_value: possibly type-converted value

    Raises:
        TypeError: if value and dtype mismatch
    """
    if isinstance(dtype_or_dtype_tuple, (tuple, list)):
        if convert_func is None:
            convert_func = dtype_or_dtype_tuple[0]

        valid_types = tuple(dtype_or_dtype_tuple)
        if not isinstance(value, valid_types):
            raise TypeError(f"Expect type in {valid_types}, got {type(value)}: {value}")
        if convert:
            return convert_func(value)
        else:
            return value

    else:
        valid_type = dtype_or_dtype_tuple
        if not isinstance(value, valid_type):
            raise TypeError(f"Expect type {valid_type}, got {type(value)}: {value}")

        if convert and convert_func is not None:
            return convert_func(value)
        else:
            return value


def assure_values_type(values, dtype_or_dtype_tuple, convert_func=None, convert=True):
    """
    List version of assure_value_type.
    If values is None, return None

    Args:
        values: list of objects of built-in type
        dtype_or_dtype_tuple: a type object or a tuple of type object
        convert_func: type convert function when convert is True. Default to dtype_or_dtype_tuple[0]
                        if dtype_or_dtype_tuple is tuple
        convert: whether to convert according

    Returns:
        converted_values: possibly type-converted list of values

    Raises:
        TypeError: if value and dtype mismatch
    """
    if values is None:
        return None
    else:
        values = assure_value_type(values, (list, Iterable))
        values = [assure_value_type(value, dtype_or_dtype_tuple, convert_func=convert_func, convert=convert)
                    for value in values]
        # avoid repetition
        values = list(set(values))
    values.sort()
    return values

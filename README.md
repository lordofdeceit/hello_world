Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@lordofdeceit 
57
4,157 182 cool-RR/PySnooper
 Code  Issues 13  Pull requests 2  Projects 0  Wiki  Insights
Work feverishly

 master
@cool-RR
cool-RR committed 2 days ago
1 parent cb2d96d commit 443b442b1daf1ad42771555881e669f923604ef4
Showing  with 476 additions and 259 deletions.
  
4  .gitignore
@@ -4,4 +4,6 @@ __pycache__

.pytest_cache

*.wpu 
*.wpu

*.bak 
  
21  .travis.yml
@@ -0,0 +1,21 @@
dist: xenial
language: python
python:
#- pypy2.7
- 2.7
- 3.4
- 3.5
- 3.6
- 3.7
- 3.8-dev
- pypy3.5

env:
 - PYTHONWARNINGS='ignore::DeprecationWarning' # Until python_toolbox is fixed

install:
- pip install -r requirements.txt -r test_requirements.txt
script:
- pytest


   
26  README.md
@@ -1,6 +1,10 @@
# WORK-IN-PROGRESS, NOT USABLE YET #

# PySnooper - Never use print for debugging again #

**PySnooper** is a poor man's debugger.
[![Travis CI](https://img.shields.io/travis/cool-RR/PySnooper/master.svg)](https://travis-ci.org/cool-RR/PySnooper)

**PySnooper** is a poor man's debugger. 

You're trying to figure out why your Python code isn't doing what you think it should be doing. You'd love to use a full-fledged debugger with breakpoints and watches, but you can't be bothered to set one up right now.

@@ -33,28 +37,28 @@ We're writing a function that converts a number to binary, by returing a list of

The output to stderr is: 

                ==> number = 6
    ............... number = 6
    00:24:15.284000 call         3 @pysnooper.snoop()
    00:24:15.284000 line         5     if number:
    00:24:15.284000 line         6         bits = []
                ==> bits = []
    ............... bits = []
    00:24:15.284000 line         7         while number:
    00:24:15.284000 line         8             number, remainder = divmod(number, 2)
                ==> number = 3
                ==> remainder = 0
    ............... number = 3
    ............... remainder = 0
    00:24:15.284000 line         9             bits.insert(0, remainder)
                ==> bits = [0]
    ............... bits = [0]
    00:24:15.284000 line         7         while number:
    00:24:15.284000 line         8             number, remainder = divmod(number, 2)
                ==> number = 1
                ==> remainder = 1
    ............... number = 1
    ............... remainder = 1
    00:24:15.284000 line         9             bits.insert(0, remainder)
                ==> bits = [1, 0]
    ............... bits = [1, 0]
    00:24:15.284000 line         7         while number:
    00:24:15.284000 line         8             number, remainder = divmod(number, 2)
                ==> number = 0
    ............... number = 0
    00:24:15.284000 line         9             bits.insert(0, remainder)
                ==> bits = [1, 1, 0]
    ............... bits = [1, 1, 0]
    00:24:15.284000 line         7         while number:
    00:24:15.284000 line        10         return bits
    00:24:15.284000 return      10         return bits
  
6  misc/IDE files/pysnooper.wpr → misc/IDE files/PySnooper.wpr
@@ -12,7 +12,11 @@ proj.directory-list = [{'dirloc': loc('../..'),
                        'watch_for_changes': True}]
proj.file-type = 'shared'
proj.home-dir = loc('../..')
proj.launch-config = {loc('../../../../../Dropbox/Scripts and shortcuts/_simplify3d_add_m600.py'): ('p'\
proj.launch-config = {loc('../../../../../../../Program Files/Python37/Scripts/pasteurize-script.py'): ('p'\
        'roject',
        (u'"c:\\Users\\Administrator\\Documents\\Python Projects\\PySnooper\\pysnooper" "c:\\Users\\Administrator\\Documents\\Python Projects\\PySnooper\\tests"',
         '')),
                      loc('../../../../../Dropbox/Scripts and shortcuts/_simplify3d_add_m600.py'): ('p'\
        'roject',
        (u'"C:\\Users\\Administrator\\Dropbox\\Desktop\\foo.gcode"',
         ''))}
  
33  pysnooper/pycompat.py
@@ -0,0 +1,33 @@
# Copyright 2019 Ram Rachum.
# This program is distributed under the MIT license.
'''Python 2/3 compatibilty'''

import abc
import os

if hasattr(abc, 'ABC'):
    ABC = abc.ABC
else:
    class ABC(object):
        """Helper class that provides a standard way to create an ABC using
        inheritance.
        """
        __metaclass__ = abc.ABCMeta
        __slots__ = ()


if hasattr(os, 'PathLike'):
    PathLike = os.PathLike
else:
    class PathLike(ABC):
        """Abstract base class for implementing the file system path protocol."""

        @abc.abstractmethod
        def __fspath__(self):
            """Return the file system path representation of the object."""
            raise NotImplementedError

        @classmethod
        def __subclasshook__(cls, subclass):
            return hasattr(subclass, '__fspath__')

  
140  pysnooper/pysnooper.py
@@ -1,30 +1,29 @@
# Copyright 2019 Ram Rachum.
# This program is distributed under the MIT license.

from __future__ import annotations

import sys
import os
import pathlib
import inspect
import types
import typing
import datetime as datetime_module
import re
import collections

import decorator

from . import utils
from . import pycompat
from .tracer import Tracer


def get_write_function(output) -> typing.Callable:
def get_write_function(output):
    if output is None:
        def write(s):
            stderr = sys.stderr
            stderr.write(s)
            stderr.write('\n')
    elif isinstance(output, (os.PathLike, str)):
    elif isinstance(output, (pycompat.PathLike, str)):
        output_path = pathlib.Path(output)
        def write(s):
            with output_path.open('a') as output_file:
@@ -39,136 +38,15 @@ def write(s):
    return write


class Tracer:
    def __init__(self, target_code_object: types.CodeType, write: callable, *,
                 variables: typing.Sequence=()):
        self.target_code_object = target_code_object
        self.write = write
        self.variables = variables
        self.old_local_reprs = {}
        self.local_reprs = {}


    def __enter__(self):
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.trace)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        sys.settrace(self.original_trace_function)

    def trace(self: Tracer, frame: types.FrameType, event: str,
              arg: typing.Any) -> typing.Callable:
        if frame.f_code != self.target_code_object:
            return self.trace
        self.old_local_reprs, self.local_reprs = \
             self.local_reprs, get_local_reprs(frame, variables=self.variables)
        modified_local_reprs = {
            key: value for key, value in self.local_reprs.items()
            if (key not in self.old_local_reprs) or
                                           (self.old_local_reprs[key] != value)
        }
        for name, value_repr in modified_local_reprs.items():
            self.write(f'            ==> {name} = {value_repr}')
        # x = repr((frame.f_code.co_stacksize, frame, event, arg))
        now_string = datetime_module.datetime.now().time().isoformat()
        source_line = get_source_from_frame(frame)[frame.f_lineno - 1]
        self.write(f'{now_string} {event:9} '
                   f'{frame.f_lineno:4} {source_line}')
        return self.trace



source_cache_by_module_name = {}
source_cache_by_file_name = {}
def get_source_from_frame(frame: types.FrameType) -> str:
    module_name = frame.f_globals.get('__name__') or ''
    if module_name:
        try:
            return source_cache_by_module_name[module_name]
        except KeyError:
            pass
    file_name = frame.f_code.co_filename
    if file_name:
        try:
            return source_cache_by_file_name[file_name]
        except KeyError:
            pass
    function = frame.f_code.co_name
    loader = frame.f_globals.get('__loader__')

    source: typing.Union[None, str] = None
    if hasattr(loader, 'get_source'):
        try:
            source = loader.get_source(module_name)
        except ImportError:
            pass
        if source is not None:
            source = source.splitlines()
    if source is None:
        try:
            with open(file_name, 'rb') as fp:
                source = fp.read().splitlines()
        except (OSError, IOError):
            pass
    if source is None:
        raise NotImplementedError

    # If we just read the source from a file, or if the loader did not
    # apply tokenize.detect_encoding to decode the source into a
    # string, then we should do that ourselves.
    if isinstance(source[0], bytes):
        encoding = 'ascii'
        for line in source[:2]:
            # File coding may be specified. Match pattern from PEP-263
            # (https://www.python.org/dev/peps/pep-0263/)
            match = re.search(br'coding[:=]\s*([-\w.]+)', line)
            if match:
                encoding = match.group(1).decode('ascii')
                break
        source = [str(sline, encoding, 'replace') for sline in source]

    if module_name:
        source_cache_by_module_name[module_name] = source
    if file_name:
        source_cache_by_file_name[file_name] = source
    return source

def get_local_reprs(frame: types.FrameType, *, variables: typing.Sequence=()) -> dict:
    result = {}
    for key, value in frame.f_locals.items():
        try:
            result[key] = get_shortish_repr(value)
        except Exception:
            continue
    locals_and_globals = collections.ChainMap(frame.f_locals, frame.f_globals)
    for variable in variables:
        steps = variable.split('.')
        step_iterator = iter(steps)
        try:
            current = locals_and_globals[next(step_iterator)]
            for step in step_iterator:
                current = getattr(current, step)
        except (KeyError, AttributeError):
            continue
        try:
            result[variable] = get_shortish_repr(current)
        except Exception:
            continue
    return result

def get_shortish_repr(item) -> str:
    r = repr(item)
    if len(r) > 100:
        r = f'{r[:97]}...'
    return r


def snoop(output=None, *, variables=()) -> typing.Callable:
def snoop(output=None, *, variables=(), depth=1):
    write = get_write_function(output)
    @decorator.decorator
    def decorate(function, *args, **kwargs) -> typing.Callable:
    def decorate(function, *args, **kwargs):
        target_code_object = function.__code__
        with Tracer(target_code_object, write, variables=variables):
        with Tracer(target_code_object=target_code_object,
                    write=write, variables=variables,
                    depth=depth):
            return function(*args, **kwargs)

    return decorate
  
179  pysnooper/tracer.py
@@ -0,0 +1,179 @@
# Copyright 2019 Ram Rachum.
# This program is distributed under the MIT license.

import types
import sys
import re
import collections
import datetime as datetime_module

def get_shortish_repr(item) -> str:
    r = repr(item)
    if len(r) > 100:
        r = '{r[:97]}...'.format(**locals())
    return r

def get_local_reprs(frame: types.FrameType, *,
                    variables=()) -> dict:
    result = {}
    for key, value in frame.f_locals.items():
        try:
            result[key] = get_shortish_repr(value)
        except Exception:
            continue
    locals_and_globals = collections.ChainMap(frame.f_locals, frame.f_globals)
    for variable in variables:
        steps = variable.split('.')
        step_iterator = iter(steps)
        try:
            current = locals_and_globals[next(step_iterator)]
            for step in step_iterator:
                current = getattr(current, step)
        except (KeyError, AttributeError):
            continue
        try:
            result[variable] = get_shortish_repr(current)
        except Exception:
            continue
    return result


source_cache_by_module_name = {}
source_cache_by_file_name = {}
def get_source_from_frame(frame: types.FrameType) -> str:
    module_name = frame.f_globals.get('__name__') or ''
    if module_name:
        try:
            return source_cache_by_module_name[module_name]
        except KeyError:
            pass
    file_name = frame.f_code.co_filename
    if file_name:
        try:
            return source_cache_by_file_name[file_name]
        except KeyError:
            pass
    function = frame.f_code.co_name
    loader = frame.f_globals.get('__loader__')

    source = None
    if hasattr(loader, 'get_source'):
        try:
            source = loader.get_source(module_name)
        except ImportError:
            pass
        if source is not None:
            source = source.splitlines()
    if source is None:
        try:
            with open(file_name, 'rb') as fp:
                source = fp.read().splitlines()
        except (OSError, IOError):
            pass
    if source is None:
        raise NotImplementedError

    # If we just read the source from a file, or if the loader did not
    # apply tokenize.detect_encoding to decode the source into a
    # string, then we should do that ourselves.
    if isinstance(source[0], bytes):
        encoding = 'ascii'
        for line in source[:2]:
            # File coding may be specified. Match pattern from PEP-263
            # (https://www.python.org/dev/peps/pep-0263/)
            match = re.search(br'coding[:=]\s*([-\w.]+)', line)
            if match:
                encoding = match.group(1).decode('ascii')
                break
        source = [str(sline, encoding, 'replace') for sline in source]

    if module_name:
        source_cache_by_module_name[module_name] = source
    if file_name:
        source_cache_by_file_name[file_name] = source
    return source

class Tracer:
    def __init__(self, *, target_code_object: types.CodeType, write: callable,
                 variables=(), depth: int=1):
        self.target_code_object = target_code_object
        self.write = write
        self.variables = variables
        self.frame_to_old_local_reprs = collections.defaultdict(lambda: {})
        self.frame_to_local_reprs = collections.defaultdict(lambda: {})
        self.depth = depth
        assert self.depth >= 1

    def __enter__(self):
        self.original_trace_function = sys.gettrace()
        sys.settrace(self.trace)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        sys.settrace(self.original_trace_function)


    def trace(self: 'Tracer', frame: types.FrameType, event: str,
              arg):

        ### Checking whether we should trace this line: #######################
        #                                                                     #
        # We should trace this line either if it's in the decorated function,
        # or the user asked to go a few levels deeper and we're within that
        # number of levels deeper.

        if frame.f_code is not self.target_code_object:
            if self.depth == 1:
                # We did the most common and quickest check above, because the
                # trace function runs so incredibly often, therefore it's
                # crucial to hyper-optimize it for the common case.
                return self.trace
            else:
                _frame_candidate = frame
                for i in range(1, self.depth):
                    _frame_candidate = _frame_candidate.f_back
                    if _frame_candidate is None:
                        return self.trace
                    elif _frame_candidate.f_code is self.target_code_object:
                        indent = ' ' * 4 * i
                        break
                else:
                    return self.trace
        else:
            indent = ''
        #                                                                     #
        ### Finished checking whether we should trace this line. ##############

        ### Reporting newish and modified variables: ##########################
        #                                                                     #
        self.frame_to_old_local_reprs[frame] = old_local_reprs = \
                                               self.frame_to_local_reprs[frame]
        self.frame_to_local_reprs[frame] = local_reprs = \
                               get_local_reprs(frame, variables=self.variables)

        modified_local_reprs = {}
        newish_local_reprs = {}

        for key, value in local_reprs.items():
            if key not in old_local_reprs:
                newish_local_reprs[key] = value
            elif old_local_reprs[key] != value:
                modified_local_reprs[key] = value

        newish_string = ('Starting var:.. ' if event == 'call' else
                                                            'New var:....... ')
        for name, value_repr in newish_local_reprs.items():
            self.write('{indent}{newish_string}{name} = {value_repr}'.format(
                                                                   **locals()))
        for name, value_repr in modified_local_reprs.items():
            self.write('{indent}Modified var:.. {name} = {value_repr}'.format(
                                                                   **locals()))
        #                                                                     #
        ### Finished newish and modified variables. ###########################

        now_string = datetime_module.datetime.now().time().isoformat()
        source_line = get_source_from_frame(frame)[frame.f_lineno - 1]
        self.write('{indent}{now_string} {event:9} '
                   '{frame.f_lineno:4} {source_line}'.format(**locals()))
        return self.trace


  
3  pysnooper/utils.py
@@ -4,6 +4,7 @@
import abc
import sys

from .pycompat import ABC

def _check_methods(C, *methods):
    mro = C.__mro__
@@ -18,7 +19,7 @@ def _check_methods(C, *methods):
    return True


class WritableStream(metaclass=abc.ABCMeta):
class WritableStream(ABC):
    @abc.abstractmethod
    def write(self, s):
        pass
  
2  requirements.txt
@@ -0,0 +1,2 @@
decorator>=4.3.0
future>=0.17.1
  
2  test_requirements.txt
@@ -0,0 +1,2 @@
python_toolbox>=0.9.3
pytest>=4.4.1
  
181  tests/test_pysnooper.py
@@ -10,114 +10,11 @@

import pysnooper

from .utils import (assert_output, VariableEntry, CallEntry, LineEntry,
                    ReturnEntry, OpcodeEntry, ExceptionEntry)

class Entry(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def check(self, s: str) -> bool:
        pass

class VariableEntry(Entry):
    line_pattern = re.compile(
        r"""^            ==> (?P<name>[^ ]*) = (?P<value>.*)$"""
    )
    def __init__(self, name=None, value=None, *,
                 name_regex=None, value_regex=None):
        if name is not None:
            assert name_regex is None
        if value is not None:
            assert value_regex is None

        self.name = name
        self.value = value
        self.name_regex = (None if name_regex is None else
                                                        re.compile(name_regex))
        self.value_regex = (None if value_regex is None else
                                                       re.compile(value_regex))

    def _check_name(self, name: str) -> bool:
        if self.name is not None:
            return name == self.name
        elif self.name_regex is not None:
            return self.name_regex.fullmatch(name)
        else:
            return True

    def _check_value(self, value: str) -> bool:
        if self.value is not None:
            return value == self.value
        elif self.value_regex is not None:
            return self.value_regex.fullmatch(value)
        else:
            return True

    def check(self, s: str) -> bool:
        match: re.Match = self.line_pattern.fullmatch(s)
        if not match:
            return False
        name, value = match.groups()
        return self._check_name(name) and self._check_value(value)


class EventEntry(Entry):
    def __init__(self, source=None, *, source_regex=None):
        if source is not None:
            assert source_regex is None

        self.source = source
        self.source_regex = (None if source_regex is None else
                                                      re.compile(source_regex))

    line_pattern = re.compile(
        (r"""^[0-9:.]{15} (?P<event_name>[a-z]*) +"""
         r"""(?P<line_number>[0-9]*) +(?P<source>.*)$""")
    )

    @caching.CachedProperty
    def event_name(self):
        return re.match('^[A-Z][a-z]*', type(self).__name__).group(0).lower()

    def _check_source(self, source: str) -> bool:
        if self.source is not None:
            return source == self.source
        elif self.source_regex is not None:
            return self.source_regex.fullmatch(source)
        else:
            return True

    def check(self, s: str) -> bool:
        match: re.Match = self.line_pattern.fullmatch(s)
        if not match:
            return False
        event_name, _, source = match.groups()
        return event_name == self.event_name and self._check_source(source)



class CallEntry(EventEntry):
    pass

class LineEntry(EventEntry):
    pass

class ReturnEntry(EventEntry):
    pass

class ExceptionEntry(EventEntry):
    pass

class OpcodeEntry(EventEntry):
    pass


def check_output(output, expected_entries):
    lines = tuple(filter(None, output.split('\n')))
    if len(lines) != len(expected_entries):
        return False
    return all(expected_entry.check(line) for
               expected_entry, line in zip(expected_entries, lines))


def test_pysnooper():
def test_string_io():
    string_io = io.StringIO()
    @pysnooper.snoop(string_io)
    def my_function(foo):
@@ -127,10 +24,10 @@ def my_function(foo):
    result = my_function('baba')
    assert result == 15
    output = string_io.getvalue()
    assert check_output(
    assert_output(
        output,
        (
            VariableEntry('foo', "'baba'"),
            VariableEntry('foo', value_regex="u?'baba'"),
            CallEntry(),
            LineEntry('x = 7'), 
            VariableEntry('x', '7'), 
@@ -161,15 +58,15 @@ def my_function():
        result = my_function()
    assert result is None
    output = output_capturer.string_io.getvalue()
    assert check_output(
    assert_output(
        output,
        (
            VariableEntry('Foo'),
            VariableEntry('re'),
            VariableEntry(),
            VariableEntry(),
            CallEntry(),
            LineEntry('foo = Foo()'), 
            VariableEntry('foo'),
            VariableEntry('foo.x', '2'),
            VariableEntry(),
            VariableEntry(),
            LineEntry(), 
            VariableEntry('i', '0'),
            LineEntry(), 
@@ -181,4 +78,60 @@ def my_function():
            LineEntry(), 
            ReturnEntry(), 
        )
    ) 
    )

def test_depth():
    string_io = io.StringIO()

    def f4(x4):
        result4 = x4 * 2
        return result4

    def f3(x3):
        result3 = f4(x3)
        return result3

    def f2(x2):
        result2 = f3(x2)
        return result2

    @pysnooper.snoop(string_io, depth=3)
    def f1(x1):
        result1 = f2(x1)
        return result1

    result = f1(10)
    assert result == 20
    output = string_io.getvalue()
    assert_output(
        output,
        (
            VariableEntry(),
            VariableEntry(),
            CallEntry(),
            LineEntry(), 

            VariableEntry(),
            VariableEntry(),
            CallEntry(),
            LineEntry(), 

            VariableEntry(),
            VariableEntry(),
            CallEntry(),
            LineEntry(),

            VariableEntry(),
            LineEntry(),
            ReturnEntry(),

            VariableEntry(),
            LineEntry(),
            ReturnEntry(),

            VariableEntry(),
            LineEntry(),
            ReturnEntry(),
        )
    )

  
138  tests/utils.py
@@ -0,0 +1,138 @@
# Copyright 2019 Ram Rachum.
# This program is distributed under the MIT license.

import re
import abc

from python_toolbox import caching

import pysnooper.pycompat


class _BaseEntry(pysnooper.pycompat.ABC):
    @abc.abstractmethod
    def check(self, s: str) -> bool:
        pass

class VariableEntry(_BaseEntry):
    line_pattern = re.compile(
        r"""^(?P<indent>(?: {4})*)(?P<stage>New|Modified|Starting) var:"""
        r"""\.{2,7} (?P<name>[^ ]+) = (?P<value>.+)$"""
    )
    def __init__(self, name=None, value=None, stage=None, *,
                 name_regex=None, value_regex=None):
        if name is not None:
            assert name_regex is None
        if value is not None:
            assert value_regex is None
        assert stage in (None, 'starting', 'new', 'modified')

        self.name = name
        self.value = value
        self.stage = stage
        self.name_regex = (None if name_regex is None else
                           re.compile(name_regex))
        self.value_regex = (None if value_regex is None else
                            re.compile(value_regex))

    def _check_name(self, name: str) -> bool:
        if self.name is not None:
            return name == self.name
        elif self.name_regex is not None:
            return self.name_regex.match(name)
        else:
            return True

    def _check_value(self, value: str) -> bool:
        if self.value is not None:
            return value == self.value
        elif self.value_regex is not None:
            return self.value_regex.match(value)
        else:
            return True

    def _check_stage(self, stage: str) -> bool:
        stage = stage.lower()
        if self.stage is None:
            return stage in ('starting', 'new', 'modified')
        else:
            return stage == self.value

    def check(self, s: str) -> bool:
        match = self.line_pattern.match(s)
        if not match:
            return False
        indent, stage, name, value = match.groups()
        return (self._check_name(name) and self._check_value(value) and
                self._check_stage(stage))


class _BaseEventEntry(_BaseEntry):
    def __init__(self, source=None, *, source_regex=None):
        if type(self) is _BaseEventEntry:
            raise TypeError
        if source is not None:
            assert source_regex is None

        self.source = source
        self.source_regex = (None if source_regex is None else
                             re.compile(source_regex))

    line_pattern = re.compile(
        (r"""^(?P<indent>(?: {4})*)[0-9:.]{15} (?P<event_name>[a-z]*) +"""
         r"""(?P<line_number>[0-9]*) +(?P<source>.*)$""")
    )

    @caching.CachedProperty
    def event_name(self):
        return re.match('^[A-Z][a-z]*', type(self).__name__).group(0).lower()

    def _check_source(self, source: str) -> bool:
        if self.source is not None:
            return source == self.source
        elif self.source_regex is not None:
            return self.source_regex.match(source)
        else:
            return True

    def check(self, s: str) -> bool:
        match = self.line_pattern.match(s)
        if not match:
            return False
        indent, event_name, _, source = match.groups()
        return event_name == self.event_name and self._check_source(source)



class CallEntry(_BaseEventEntry):
    pass

class LineEntry(_BaseEventEntry):
    pass

class ReturnEntry(_BaseEventEntry):
    pass

class ExceptionEntry(_BaseEventEntry):
    pass

class OpcodeEntry(_BaseEventEntry):
    pass


class OutputFailure(Exception):
    pass


def assert_output(output, expected_entries):
    lines = tuple(filter(None, output.split('\n')))
    if len(lines) != len(expected_entries):
        raise OutputFailure(
            'Output has {len(lines)} lines, while we expect '
            '{len(expected_entries)} lines.'.format(**locals())
        )
    for expected_entry, line in zip(expected_entries, lines):
        if not expected_entry.check(line):
            raise OutputFailure(line)


0 comments on commit 443b442
@lordofdeceit
   
 
 
 
Leave a comment

Attach files by dragging & dropping, selecting or pasting them.
  You’re not receiving notifications from this thread.
© 2019 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x4a2c247c

# Compiled with Coconut version 2.0.0-a_dev33 [How Not to Be Seen]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop, annotations
import sys as _coconut_sys
from builtins import chr, filter, hex, input, int, map, object, oct, open, print, range, str, zip, filter, reversed, enumerate
py_chr, py_hex, py_input, py_int, py_map, py_object, py_oct, py_open, py_print, py_range, py_str, py_zip, py_filter, py_reversed, py_enumerate, py_repr = chr, hex, input, int, map, object, oct, open, print, range, str, zip, filter, reversed, enumerate, repr
_coconut_py_str = str
exec("_coconut_exec = exec")
py_breakpoint = breakpoint
class _coconut:
    import collections, copy, functools, types, itertools, operator, threading, os, warnings, contextlib, traceback, weakref, multiprocessing, math
    from multiprocessing import dummy as multiprocessing_dummy
    import asyncio
    import pickle
    OrderedDict = collections.OrderedDict
    import collections.abc as abc
    import typing
    zip_longest = itertools.zip_longest
    try:
        import numpy
    except ImportError:
        class you_need_to_install_numpy: pass
        numpy = you_need_to_install_numpy()
    else:
        abc.Sequence.register(numpy.ndarray)
    abc.Sequence.register(collections.deque)
    Ellipsis, NotImplemented, NotImplementedError, Exception, AttributeError, ImportError, IndexError, KeyError, NameError, TypeError, ValueError, StopIteration, RuntimeError, all, any, bytes, classmethod, dict, enumerate, filter, float, frozenset, getattr, hasattr, hash, id, int, isinstance, issubclass, iter, len, list, locals, map, min, max, next, object, property, range, reversed, set, slice, str, sum, super, tuple, type, vars, zip, repr, print = Ellipsis, NotImplemented, NotImplementedError, Exception, AttributeError, ImportError, IndexError, KeyError, NameError, TypeError, ValueError, StopIteration, RuntimeError, all, any, bytes, classmethod, dict, enumerate, filter, float, frozenset, getattr, hasattr, hash, id, int, isinstance, issubclass, iter, len, list, locals, map, min, max, next, object, property, range, reversed, set, slice, str, sum, super, tuple, type, vars, zip, repr, print
class _coconut_sentinel: pass
class _coconut_base_hashable:
    __slots__ = ()
    def __reduce_ex__(self, _):
        return self.__reduce__()
    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.__reduce__() == other.__reduce__()
    def __hash__(self):
        return _coconut.hash(self.__reduce__())
class MatchError(_coconut_base_hashable, Exception):
    """Pattern-matching error. Has attributes .pattern, .value, and .message."""
    __slots__ = ("pattern", "value", "_message")
    max_val_repr_len = 500
    def __init__(self, pattern=None, value=None):
        self.pattern = pattern
        self.value = value
        self._message = None
    @property
    def message(self):
        if self._message is None:
            value_repr = _coconut.repr(self.value)
            self._message = "pattern-matching failed for %s in %s" % (_coconut.repr(self.pattern), value_repr if _coconut.len(value_repr) <= self.max_val_repr_len else value_repr[:self.max_val_repr_len] + "...")
            Exception.__init__(self, self._message)
        return self._message
    def __repr__(self):
        self.message
        return Exception.__repr__(self)
    def __str__(self):
        self.message
        return Exception.__str__(self)
    def __unicode__(self):
        self.message
        return Exception.__unicode__(self)
    def __reduce__(self):
        return (self.__class__, (self.pattern, self.value))
class _coconut_tail_call:
    __slots__ = ("func", "args", "kwargs")
    def __init__(self, _coconut_func, *args, **kwargs):
        self.func = _coconut_func
        self.args = args
        self.kwargs = kwargs
_coconut_tco_func_dict = {}
def _coconut_tco(func):
    @_coconut.functools.wraps(func)
    def tail_call_optimized_func(*args, **kwargs):
        call_func = func
        while True:
            if _coconut.isinstance(call_func, _coconut_base_pattern_func):
                call_func = call_func._coconut_tco_func
            elif _coconut.isinstance(call_func, _coconut.types.MethodType):
                wkref = _coconut_tco_func_dict.get(_coconut.id(call_func.__func__))
                wkref_func = None if wkref is None else wkref()
                if wkref_func is call_func.__func__:
                    if call_func.__self__ is None:
                        call_func = call_func._coconut_tco_func
                    else:
                        call_func = _coconut.functools.partial(call_func._coconut_tco_func, call_func.__self__)
            else:
                wkref = _coconut_tco_func_dict.get(_coconut.id(call_func))
                wkref_func = None if wkref is None else wkref()
                if wkref_func is call_func:
                    call_func = call_func._coconut_tco_func
            result = call_func(*args, **kwargs)  # pass --no-tco to clean up your traceback
            if not isinstance(result, _coconut_tail_call):
                return result
            call_func, args, kwargs = result.func, result.args, result.kwargs
    tail_call_optimized_func._coconut_tco_func = func
    tail_call_optimized_func.__module__ = _coconut.getattr(func, "__module__", None)
    tail_call_optimized_func.__name__ = _coconut.getattr(func, "__name__", None)
    tail_call_optimized_func.__qualname__ = _coconut.getattr(func, "__qualname__", None)
    _coconut_tco_func_dict[_coconut.id(tail_call_optimized_func)] = _coconut.weakref.ref(tail_call_optimized_func)
    return tail_call_optimized_func
def _coconut_iter_getitem_special_case(iterable, start, stop, step):
    iterable = _coconut.itertools.islice(iterable, start, None)
    cache = _coconut.collections.deque(_coconut.itertools.islice(iterable, -stop), maxlen=-stop)
    for index, item in _coconut.enumerate(iterable):
        cached_item = cache.popleft()
        if index % step == 0:
            yield cached_item
        cache.append(item)
def _coconut_iter_getitem(iterable, index):
    """Some code taken from more_itertools under the terms of its MIT license."""
    obj_iter_getitem = _coconut.getattr(iterable, "__iter_getitem__", None)
    if obj_iter_getitem is None:
        obj_iter_getitem = _coconut.getattr(iterable, "__getitem__", None)
    if obj_iter_getitem is not None:
        try:
            result = obj_iter_getitem(index)
        except _coconut.NotImplementedError:
            pass
        else:
            if result is not _coconut.NotImplemented:
                return result
    if not _coconut.isinstance(index, _coconut.slice):
        if index < 0:
            return _coconut.collections.deque(iterable, maxlen=-index)[0]
        result = _coconut.next(_coconut.itertools.islice(iterable, index, index + 1), _coconut_sentinel)
        if result is _coconut_sentinel:
            raise _coconut.IndexError("$[] index out of range")
        return result
    start, stop, step = index.start, index.stop, 1 if index.step is None else index.step
    if step == 0:
        raise _coconut.ValueError("slice step cannot be zero")
    if start is None and stop is None and step == -1:
        obj_reversed = _coconut.getattr(iterable, "__reversed__", None)
        if obj_reversed is not None:
            try:
                result = obj_reversed()
            except _coconut.NotImplementedError:
                pass
            else:
                if result is not _coconut.NotImplemented:
                    return result
    if step >= 0:
        start = 0 if start is None else start
        if start < 0:
            cache = _coconut.collections.deque(_coconut.enumerate(iterable, 1), maxlen=-start)
            len_iter = cache[-1][0] if cache else 0
            i = _coconut.max(len_iter + start, 0)
            if stop is None:
                j = len_iter
            elif stop >= 0:
                j = _coconut.min(stop, len_iter)
            else:
                j = _coconut.max(len_iter + stop, 0)
            n = j - i
            if n <= 0:
                return ()
            if n < -start or step != 1:
                cache = _coconut.itertools.islice(cache, 0, n, step)
            return _coconut_map(_coconut.operator.itemgetter(1), cache)
        elif stop is None or stop >= 0:
            return _coconut.itertools.islice(iterable, start, stop, step)
        else:
            return _coconut_iter_getitem_special_case(iterable, start, stop, step)
    else:
        start = -1 if start is None else start
        if stop is not None and stop < 0:
            n = -stop - 1
            cache = _coconut.collections.deque(_coconut.enumerate(iterable, 1), maxlen=n)
            len_iter = cache[-1][0] if cache else 0
            if start < 0:
                i, j = start, stop
            else:
                i, j = _coconut.min(start - len_iter, -1), None
            return _coconut_map(_coconut.operator.itemgetter(1), _coconut.tuple(cache)[i:j:step])
        else:
            if stop is not None:
                m = stop + 1
                iterable = _coconut.itertools.islice(iterable, m, None)
            if start < 0:
                i = start
                n = None
            elif stop is None:
                i = None
                n = start + 1
            else:
                i = None
                n = start - stop
            if n is not None:
                if n <= 0:
                    return ()
                iterable = _coconut.itertools.islice(iterable, 0, n)
            return _coconut.tuple(iterable)[i::step]
class _coconut_base_compose(_coconut_base_hashable):
    __slots__ = ("func", "funcstars")
    def __init__(self, func, *funcstars):
        self.func = func
        self.funcstars = []
        for f, stars in funcstars:
            if _coconut.isinstance(f, _coconut_base_compose):
                self.funcstars.append((f.func, stars))
                self.funcstars += f.funcstars
            else:
                self.funcstars.append((f, stars))
        self.funcstars = _coconut.tuple(self.funcstars)
    def __call__(self, *args, **kwargs):
        arg = self.func(*args, **kwargs)
        for f, stars in self.funcstars:
            if stars == 0:
                arg = f(arg)
            elif stars == 1:
                arg = f(*arg)
            elif stars == 2:
                arg = f(**arg)
            else:
                raise _coconut.ValueError("invalid arguments to " + _coconut.repr(self))
        return arg
    def __repr__(self):
        return _coconut.repr(self.func) + " " + " ".join(("..*> " if star == 1 else "..**>" if star == 2 else "..> ") + _coconut.repr(f) for f, star in self.funcstars)
    def __reduce__(self):
        return (self.__class__, (self.func,) + self.funcstars)
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _coconut.types.MethodType(self, obj)
def _coconut_forward_compose(func, *funcs): return _coconut_base_compose(func, *((f, 0) for f in funcs))
def _coconut_back_compose(*funcs): return _coconut_forward_compose(*_coconut.reversed(funcs))
def _coconut_forward_star_compose(func, *funcs): return _coconut_base_compose(func, *((f, 1) for f in funcs))
def _coconut_back_star_compose(*funcs): return _coconut_forward_star_compose(*_coconut.reversed(funcs))
def _coconut_forward_dubstar_compose(func, *funcs): return _coconut_base_compose(func, *((f, 2) for f in funcs))
def _coconut_back_dubstar_compose(*funcs): return _coconut_forward_dubstar_compose(*_coconut.reversed(funcs))
def _coconut_pipe(x, f): return f(x)
def _coconut_star_pipe(xs, f): return f(*xs)
def _coconut_dubstar_pipe(kws, f): return f(**kws)
def _coconut_back_pipe(f, x): return f(x)
def _coconut_back_star_pipe(f, xs): return f(*xs)
def _coconut_back_dubstar_pipe(f, kws): return f(**kws)
def _coconut_none_pipe(x, f): return None if x is None else f(x)
def _coconut_none_star_pipe(xs, f): return None if xs is None else f(*xs)
def _coconut_none_dubstar_pipe(kws, f): return None if kws is None else f(**kws)
def _coconut_assert(cond, msg=None):
    if not cond:
        assert False, msg if msg is not None else "(assert) got falsey value " + _coconut.repr(cond)
def _coconut_bool_and(a, b): return a and b
def _coconut_bool_or(a, b): return a or b
def _coconut_none_coalesce(a, b): return b if a is None else a
def _coconut_minus(a, b=_coconut_sentinel):
    if b is _coconut_sentinel:
        return -a
    return a - b
def _coconut_comma_op(*args): return args
@_coconut.functools.wraps(_coconut.itertools.tee)
def tee(iterable, n=2):
    if n >= 0 and _coconut.isinstance(iterable, (_coconut.tuple, _coconut.frozenset)):
        return (iterable,) * n
    if n > 0 and (_coconut.isinstance(iterable, _coconut.abc.Sequence) or _coconut.getattr(iterable, "__copy__", None) is not None):
        return (iterable,) + _coconut.tuple(_coconut.copy.copy(iterable) for _ in _coconut.range(n - 1))
    return _coconut.itertools.tee(iterable, n)
class reiterable(_coconut_base_hashable):
    """Allow an iterator to be iterated over multiple times with the same results."""
    __slots__ = ("lock", "iter")
    def __new__(cls, iterable):
        if _coconut.isinstance(iterable, _coconut_reiterable):
            return iterable
        self = _coconut.object.__new__(cls)
        self.lock = _coconut.threading.Lock()
        self.iter = iterable
        return self
    def get_new_iter(self):
        with self.lock:
            self.iter, new_iter = _coconut_tee(self.iter)
        return new_iter
    def __iter__(self):
        return _coconut.iter(self.get_new_iter())
    def __getitem__(self, index):
        return _coconut_iter_getitem(self.get_new_iter(), index)
    def __reversed__(self):
        return _coconut_reversed(self.get_new_iter())
    def __len__(self):
        return _coconut.len(self.iter)
    def __repr__(self):
        return "reiterable(%r)" % (self.iter,)
    def __reduce__(self):
        return (self.__class__, (self.iter,))
    def __copy__(self):
        return self.__class__(self.get_new_iter())
    def __fmap__(self, func):
        return _coconut_map(func, self)
class scan(_coconut_base_hashable):
    """Reduce func over iterable, yielding intermediate results,
    optionally starting from initial."""
    __slots__ = ("func", "iter", "initial")
    def __init__(self, function, iterable, initial=_coconut_sentinel):
        self.func = function
        self.iter = iterable
        self.initial = initial
    def __iter__(self):
        acc = self.initial
        if acc is not _coconut_sentinel:
            yield acc
        for item in self.iter:
            if acc is _coconut_sentinel:
                acc = item
            else:
                acc = self.func(acc, item)
            yield acc
    def __len__(self):
        return _coconut.len(self.iter)
    def __repr__(self):
        return "scan(%r, %s%s)" % (self.func, _coconut.repr(self.iter), "" if self.initial is _coconut_sentinel else ", " + _coconut.repr(self.initial))
    def __reduce__(self):
        return (self.__class__, (self.func, self.iter, self.initial))
    def __fmap__(self, func):
        return _coconut_map(func, self)
class reversed(_coconut_base_hashable):
    __slots__ = ("iter",)
    if hasattr(_coconut.map, "__doc__"):
        __doc__ = _coconut.reversed.__doc__
    def __new__(cls, iterable):
        if _coconut.isinstance(iterable, _coconut.range):
            return iterable[::-1]
        if not _coconut.hasattr(iterable, "__reversed__") or _coconut.isinstance(iterable, (_coconut.list, _coconut.tuple)):
            return _coconut.object.__new__(cls)
        return _coconut.reversed(iterable)
    def __init__(self, iterable):
        self.iter = iterable
    def __iter__(self):
        return _coconut.iter(_coconut.reversed(self.iter))
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return _coconut_iter_getitem(self.iter, _coconut.slice(-(index.start + 1) if index.start is not None else None, -(index.stop + 1) if index.stop else None, -(index.step if index.step is not None else 1)))
        return _coconut_iter_getitem(self.iter, -(index + 1))
    def __reversed__(self):
        return self.iter
    def __len__(self):
        return _coconut.len(self.iter)
    def __repr__(self):
        return "reversed(%s)" % (_coconut.repr(self.iter),)
    def __reduce__(self):
        return (self.__class__, (self.iter,))
    def __contains__(self, elem):
        return elem in self.iter
    def count(self, elem):
        """Count the number of times elem appears in the reversed iterable."""
        return self.iter.count(elem)
    def index(self, elem):
        """Find the index of elem in the reversed iterable."""
        return _coconut.len(self.iter) - self.iter.index(elem) - 1
    def __fmap__(self, func):
        return self.__class__(_coconut_map(func, self.iter))
class flatten(_coconut_base_hashable):
    """Flatten an iterable of iterables into a single iterable."""
    __slots__ = ("iter",)
    def __init__(self, iterable):
        self.iter = iterable
    def __iter__(self):
        return _coconut.itertools.chain.from_iterable(self.iter)
    def __reversed__(self):
        return self.__class__(_coconut_reversed(_coconut_map(_coconut_reversed, self.iter)))
    def __repr__(self):
        return "flatten(%r)" % (self.iter,)
    def __reduce__(self):
        return (self.__class__, (self.iter,))
    def __contains__(self, elem):
        self.iter, new_iter = _coconut_tee(self.iter)
        return _coconut.any(elem in it for it in new_iter)
    def count(self, elem):
        """Count the number of times elem appears in the flattened iterable."""
        self.iter, new_iter = _coconut_tee(self.iter)
        return _coconut.sum(it.count(elem) for it in new_iter)
    def index(self, elem):
        self.iter, new_iter = _coconut_tee(self.iter)
        ind = 0
        for it in new_iter:
            try:
                return ind + it.index(elem)
            except _coconut.ValueError:
                ind += _coconut.len(it)
        raise ValueError("%r not in %r" % (elem, self))
    def __fmap__(self, func):
        return self.__class__(_coconut_map(_coconut.functools.partial(_coconut_map, func), self.iter))
class map(_coconut_base_hashable, _coconut.map):
    __slots__ = ("func", "iters")
    if hasattr(_coconut.map, "__doc__"):
        __doc__ = _coconut.map.__doc__
    def __new__(cls, function, *iterables):
        new_map = _coconut.map.__new__(cls, function, *iterables)
        new_map.func = function
        new_map.iters = iterables
        return new_map
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return self.__class__(self.func, *(_coconut_iter_getitem(i, index) for i in self.iters))
        return self.func(*(_coconut_iter_getitem(i, index) for i in self.iters))
    def __reversed__(self):
        return self.__class__(self.func, *(_coconut_reversed(i) for i in self.iters))
    def __len__(self):
        return _coconut.min(_coconut.len(i) for i in self.iters)
    def __repr__(self):
        return "map(%r, %s)" % (self.func, ", ".join((_coconut.repr(i) for i in self.iters)))
    def __reduce__(self):
        return (self.__class__, (self.func,) + self.iters)
    def __iter__(self):
        return _coconut.iter(_coconut.map(self.func, *self.iters))
    def __fmap__(self, func):
        return self.__class__(_coconut_forward_compose(self.func, func), *self.iters)
class _coconut_parallel_concurrent_map_func_wrapper(_coconut_base_hashable):
    __slots__ = ("map_cls", "func", "star")
    def __init__(self, map_cls, func, star):
        self.map_cls = map_cls
        self.func = func
        self.star = star
    def __reduce__(self):
        return (self.__class__, (self.map_cls, self.func, self.star))
    def __call__(self, *args, **kwargs):
        self.map_cls.get_pool_stack().append(None)
        try:
            if self.star:
                assert _coconut.len(args) == 1, "internal parallel/concurrent map error"
                return self.func(*args[0], **kwargs)
            else:
                return self.func(*args, **kwargs)
        except:
            _coconut.print(self.map_cls.__name__ + " error:")
            _coconut.traceback.print_exc()
            raise
        finally:
            assert self.map_cls.get_pool_stack().pop() is None, "internal parallel/concurrent map error"
class _coconut_base_parallel_concurrent_map(map):
    __slots__ = ("result", "chunksize")
    @classmethod
    def get_pool_stack(cls):
        return cls.threadlocal_ns.__dict__.setdefault("pool_stack", [None])
    def __new__(cls, function, *iterables, **kwargs):
        self = _coconut_map.__new__(cls, function, *iterables)
        self.result = None
        self.chunksize = kwargs.pop("chunksize", 1)
        if kwargs:
            raise _coconut.TypeError(cls.__name__ + "() got unexpected keyword arguments " + _coconut.repr(kwargs))
        if cls.get_pool_stack()[-1] is not None:
            return self.get_list()
        return self
    @classmethod
    @_coconut.contextlib.contextmanager
    def multiple_sequential_calls(cls, max_workers=None):
        """Context manager that causes nested calls to use the same pool."""
        if cls.get_pool_stack()[-1] is None:
            cls.get_pool_stack()[-1] = cls.make_pool(max_workers)
            try:
                yield
            finally:
                cls.get_pool_stack()[-1].terminate()
                cls.get_pool_stack()[-1] = None
        else:
            yield
    def get_list(self):
        if self.result is None:
            with self.multiple_sequential_calls():
                if _coconut.len(self.iters) == 1:
                    self.result = _coconut.list(self.get_pool_stack()[-1].imap(_coconut_parallel_concurrent_map_func_wrapper(self.__class__, self.func, False), self.iters[0], self.chunksize))
                else:
                    self.result = _coconut.list(self.get_pool_stack()[-1].imap(_coconut_parallel_concurrent_map_func_wrapper(self.__class__, self.func, True), _coconut.zip(*self.iters), self.chunksize))
        return self.result
    def __iter__(self):
        return _coconut.iter(self.get_list())
class parallel_map(_coconut_base_parallel_concurrent_map):
    """Multi-process implementation of map. Requires arguments to be pickleable.

    For multiple sequential calls, use:
        with parallel_map.multiple_sequential_calls():
            ...
    """
    __slots__ = ()
    threadlocal_ns = _coconut.threading.local()
    @staticmethod
    def make_pool(max_workers=None):
        return _coconut.multiprocessing.Pool(max_workers)
    def __repr__(self):
        return "parallel_" + _coconut_map.__repr__(self)
class concurrent_map(_coconut_base_parallel_concurrent_map):
    """Multi-thread implementation of map.

    For multiple sequential calls, use:
        with concurrent_map.multiple_sequential_calls():
            ...
    """
    __slots__ = ()
    threadlocal_ns = _coconut.threading.local()
    @staticmethod
    def make_pool(max_workers=None):
        return _coconut.multiprocessing_dummy.Pool(_coconut.multiprocessing.cpu_count() * 5 if max_workers is None else max_workers)
    def __repr__(self):
        return "concurrent_" + _coconut_map.__repr__(self)
class filter(_coconut_base_hashable, _coconut.filter):
    __slots__ = ("func", "iter")
    if hasattr(_coconut.filter, "__doc__"):
        __doc__ = _coconut.filter.__doc__
    def __new__(cls, function, iterable):
        new_filter = _coconut.filter.__new__(cls, function, iterable)
        new_filter.func = function
        new_filter.iter = iterable
        return new_filter
    def __reversed__(self):
        return self.__class__(self.func, _coconut_reversed(self.iter))
    def __repr__(self):
        return "filter(%r, %s)" % (self.func, _coconut.repr(self.iter))
    def __reduce__(self):
        return (self.__class__, (self.func, self.iter))
    def __iter__(self):
        return _coconut.iter(_coconut.filter(self.func, self.iter))
    def __fmap__(self, func):
        return _coconut_map(func, self)
class zip(_coconut_base_hashable, _coconut.zip):
    __slots__ = ("iters", "strict")
    if hasattr(_coconut.zip, "__doc__"):
        __doc__ = _coconut.zip.__doc__
    def __new__(cls, *iterables, **kwargs):
        new_zip = _coconut.zip.__new__(cls, *iterables)
        new_zip.iters = iterables
        new_zip.strict = kwargs.pop("strict", False)
        if kwargs:
            raise _coconut.TypeError("zip() got unexpected keyword arguments " + _coconut.repr(kwargs))
        return new_zip
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return self.__class__(*(_coconut_iter_getitem(i, index) for i in self.iters), strict=self.strict)
        return _coconut.tuple(_coconut_iter_getitem(i, index) for i in self.iters)
    def __reversed__(self):
        return self.__class__(*(_coconut_reversed(i) for i in self.iters), strict=self.strict)
    def __len__(self):
        return _coconut.min(_coconut.len(i) for i in self.iters)
    def __repr__(self):
        return "zip(%s%s)" % (", ".join((_coconut.repr(i) for i in self.iters)), ", strict=True" if self.strict else "")
    def __reduce__(self):
        return (self.__class__, self.iters, self.strict)
    def __setstate__(self, strict):
        self.strict = strict
    def __iter__(self):
        for items in _coconut.iter(_coconut.zip_longest(*self.iters, fillvalue=_coconut_sentinel) if self.strict else _coconut.zip(*self.iters)):
            if self.strict and _coconut.any(x is _coconut_sentinel for x in items):
                raise _coconut.ValueError("zip(..., strict=True) arguments have mismatched lengths")
            yield items
    def __fmap__(self, func):
        return _coconut_map(func, self)
class zip_longest(zip):
    __slots__ = ("fillvalue",)
    if hasattr(_coconut.zip_longest, "__doc__"):
        __doc__ = (_coconut.zip_longest).__doc__
    def __new__(cls, *iterables, **kwargs):
        self = _coconut_zip.__new__(cls, *iterables, strict=False)
        self.fillvalue = kwargs.pop("fillvalue", None)
        if kwargs:
            raise _coconut.TypeError("zip_longest() got unexpected keyword arguments " + _coconut.repr(kwargs))
        return self
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            new_ind = _coconut.slice(index.start + self.__len__() if index.start is not None and index.start < 0 else index.start, index.stop + self.__len__() if index.stop is not None and index.stop < 0 else index.stop, index.step)
            return self.__class__(*(_coconut_iter_getitem(i, new_ind) for i in self.iters))
        if index < 0:
            index += self.__len__()
        result = []
        got_non_default = False
        for it in self.iters:
            try:
                result.append(_coconut_iter_getitem(it, index))
            except _coconut.IndexError:
                result.append(self.fillvalue)
            else:
                got_non_default = True
        if not got_non_default:
            raise _coconut.IndexError("zip_longest index out of range")
        return _coconut.tuple(result)
    def __len__(self):
        return _coconut.max(_coconut.len(i) for i in self.iters)
    def __repr__(self):
        return "zip_longest(%s, fillvalue=%s)" % (", ".join((_coconut.repr(i) for i in self.iters)), _coconut.repr(self.fillvalue))
    def __reduce__(self):
        return (self.__class__, self.iters, self.fillvalue)
    def __setstate__(self, fillvalue):
        self.fillvalue = fillvalue
    def __iter__(self):
        return _coconut.iter(_coconut.zip_longest(*self.iters, fillvalue=self.fillvalue))
class enumerate(_coconut_base_hashable, _coconut.enumerate):
    __slots__ = ("iter", "start")
    if hasattr(_coconut.enumerate, "__doc__"):
        __doc__ = _coconut.enumerate.__doc__
    def __new__(cls, iterable, start=0):
        new_enumerate = _coconut.enumerate.__new__(cls, iterable, start)
        new_enumerate.iter = iterable
        new_enumerate.start = start
        return new_enumerate
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return self.__class__(_coconut_iter_getitem(self.iter, index), self.start + (0 if index.start is None else index.start if index.start >= 0 else _coconut.len(self.iter) + index.start))
        return (self.start + index, _coconut_iter_getitem(self.iter, index))
    def __len__(self):
        return _coconut.len(self.iter)
    def __repr__(self):
        return "enumerate(%s, %r)" % (_coconut.repr(self.iter), self.start)
    def __reduce__(self):
        return (self.__class__, (self.iter, self.start))
    def __iter__(self):
        return _coconut.iter(_coconut.enumerate(self.iter, self.start))
    def __fmap__(self, func):
        return _coconut_map(func, self)
class count(_coconut_base_hashable):
    """count(start, step) returns an infinite iterator starting at start and increasing by step.

    If step is set to 0, count will infinitely repeat its first argument.
    """
    __slots__ = ("start", "step")
    def __init__(self, start=0, step=1):
        self.start = start
        self.step = step
    def __iter__(self):
        while True:
            yield self.start
            if self.step:
                self.start += self.step
    def __contains__(self, elem):
        if not self.step:
            return elem == self.start
        if elem < self.start:
            return False
        return (elem - self.start) % self.step == 0
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice) and (index.start is None or index.start >= 0) and (index.stop is None or index.stop >= 0):
            new_start, new_step = self.start, self.step
            if self.step and index.start is not None:
                new_start += self.step * index.start
            if self.step and index.step is not None:
                new_step *= index.step
            if index.stop is None:
                return self.__class__(new_start, new_step)
            if self.step and _coconut.isinstance(self.start, _coconut.int) and _coconut.isinstance(self.step, _coconut.int):
                return _coconut.range(new_start, self.start + self.step * index.stop, new_step)
            return _coconut_map(self.__getitem__, _coconut.range(index.start if index.start is not None else 0, index.stop, index.step if index.step is not None else 1))
        if index < 0:
            raise _coconut.IndexError("count indices must be positive")
        return self.start + self.step * index if self.step else self.start
    def count(self, elem):
        """Count the number of times elem appears in the count."""
        if not self.step:
            return _coconut.float("inf") if elem == self.start else 0
        return int(elem in self)
    def index(self, elem):
        """Find the index of elem in the count."""
        if elem not in self:
            raise _coconut.ValueError(_coconut.repr(elem) + " not in " + _coconut.repr(self))
        return (elem - self.start) // self.step if self.step else 0
    def __reversed__(self):
        if not self.step:
            return self
        raise _coconut.TypeError(_coconut.repr(self) + " object is not reversible")
    def __repr__(self):
        return "count(%s, %s)" % (_coconut.repr(self.start), _coconut.repr(self.step))
    def __reduce__(self):
        return (self.__class__, (self.start, self.step))
    def __copy__(self):
        return self.__class__(self.start, self.step)
    def __fmap__(self, func):
        return _coconut_map(func, self)
class groupsof(_coconut_base_hashable):
    """groupsof(n, iterable) splits iterable into groups of size n.

    If the length of the iterable is not divisible by n, the last group may be of size < n.
    """
    __slots__ = ("group_size", "iter")
    def __init__(self, n, iterable):
        self.iter = iterable
        try:
            self.group_size = _coconut.int(n)
        except _coconut.ValueError:
            raise _coconut.TypeError("group size must be an int; not %r" % (n,))
        if self.group_size <= 0:
            raise _coconut.ValueError("group size must be > 0; not %r" % (self.group_size,))
    def __iter__(self):
        iterator = _coconut.iter(self.iter)
        loop = True
        while loop:
            group = []
            for _ in _coconut.range(self.group_size):
                try:
                    group.append(_coconut.next(iterator))
                except _coconut.StopIteration:
                    loop = False
                    break
            if group:
                yield _coconut.tuple(group)
    def __len__(self):
        return _coconut.int(_coconut.math.ceil(_coconut.len(self.iter) / self.group_size))
    def __repr__(self):
        return "groupsof(%r)" % (self.iter,)
    def __reduce__(self):
        return (self.__class__, (self.group_size, self.iter))
    def __fmap__(self, func):
        return _coconut_map(func, self)
class recursive_iterator(_coconut_base_hashable):
    """Decorator that optimizes a recursive function that returns an iterator (e.g. a recursive generator)."""
    __slots__ = ("func", "tee_store", "backup_tee_store")
    def __init__(self, func):
        self.func = func
        self.tee_store = {}
        self.backup_tee_store = []
    def __call__(self, *args, **kwargs):
        key = (args, _coconut.frozenset(kwargs.items()))
        use_backup = False
        try:
            _coconut.hash(key)
        except _coconut.Exception:
            try:
                key = _coconut.pickle.dumps(key, -1)
            except _coconut.Exception:
                use_backup = True
        if use_backup:
            for i, (k, v) in _coconut.enumerate(self.backup_tee_store):
                if k == key:
                    to_tee, store_pos = v, i
                    break
            else:
                to_tee = self.func(*args, **kwargs)
                store_pos = None
            to_store, to_return = _coconut_tee(to_tee)
            if store_pos is None:
                self.backup_tee_store.append([key, to_store])
            else:
                self.backup_tee_store[store_pos][1] = to_store
        else:
            it = self.tee_store.get(key)
            if it is None:
                it = self.func(*args, **kwargs)
            self.tee_store[key], to_return = _coconut_tee(it)
        return to_return
    def __repr__(self):
        return "@recursive_iterator(%r)" % (self.func,)
    def __reduce__(self):
        return (self.__class__, (self.func,))
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _coconut.types.MethodType(self, obj)
class _coconut_FunctionMatchErrorContext:
    __slots__ = ("exc_class", "taken")
    threadlocal_ns = _coconut.threading.local()
    def __init__(self, exc_class):
        self.exc_class = exc_class
        self.taken = False
    @classmethod
    def get_contexts(cls):
        try:
            return cls.threadlocal_ns.contexts
        except _coconut.AttributeError:
            cls.threadlocal_ns.contexts = []
            return cls.threadlocal_ns.contexts
    def __enter__(self):
        self.get_contexts().append(self)
    def __exit__(self, type, value, traceback):
        self.get_contexts().pop()
def _coconut_get_function_match_error():
    try:
        ctx = _coconut_FunctionMatchErrorContext.get_contexts()[-1]
    except _coconut.IndexError:
        return _coconut_MatchError
    if ctx.taken:
        return _coconut_MatchError
    ctx.taken = True
    return ctx.exc_class
class _coconut_base_pattern_func(_coconut_base_hashable):
    __slots__ = ("FunctionMatchError", "patterns", "__doc__", "__name__", "__qualname__")
    _coconut_is_match = True
    def __init__(self, *funcs):
        self.FunctionMatchError = _coconut.type(_coconut_py_str("MatchError"), (_coconut_MatchError,), {})
        self.patterns = []
        self.__doc__ = None
        self.__name__ = None
        self.__qualname__ = None
        for func in funcs:
            self.add_pattern(func)
    def add_pattern(self, func):
        if _coconut.isinstance(func, _coconut_base_pattern_func):
            self.patterns += func.patterns
        else:
            self.patterns.append(func)
        self.__doc__ = _coconut.getattr(func, "__doc__", self.__doc__)
        self.__name__ = _coconut.getattr(func, "__name__", self.__name__)
        self.__qualname__ = _coconut.getattr(func, "__qualname__", self.__qualname__)
    def __call__(self, *args, **kwargs):
        for func in self.patterns[:-1]:
            try:
                with _coconut_FunctionMatchErrorContext(self.FunctionMatchError):
                    return func(*args, **kwargs)
            except self.FunctionMatchError:
                pass
        return self.patterns[-1](*args, **kwargs)
    def _coconut_tco_func(self, *args, **kwargs):
        for func in self.patterns[:-1]:
            try:
                with _coconut_FunctionMatchErrorContext(self.FunctionMatchError):
                    return func(*args, **kwargs)
            except self.FunctionMatchError:
                pass
        return _coconut_tail_call(self.patterns[-1], *args, **kwargs)
    def __repr__(self):
        return "addpattern(%r)(*%r)" % (self.patterns[0], self.patterns[1:])
    def __reduce__(self):
        return (self.__class__, _coconut.tuple(self.patterns))
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _coconut.types.MethodType(self, obj)
def _coconut_mark_as_match(base_func):
    base_func._coconut_is_match = True
    return base_func
def addpattern(base_func, **kwargs):
    """Decorator to add a new case to a pattern-matching function (where the new case is checked last)."""
    allow_any_func = kwargs.pop("allow_any_func", False)
    if not allow_any_func and not _coconut.getattr(base_func, "_coconut_is_match", False):
        _coconut.warnings.warn("Possible misuse of addpattern with non-pattern-matching function " + _coconut.repr(base_func) + " (pass allow_any_func=True to dismiss)", stacklevel=2)
    if kwargs:
        raise _coconut.TypeError("addpattern() got unexpected keyword arguments " + _coconut.repr(kwargs))
    return _coconut.functools.partial(_coconut_base_pattern_func, base_func)
_coconut_addpattern = addpattern
def prepattern(base_func, **kwargs):
    """DEPRECATED: use addpattern instead."""
    def pattern_prepender(func):
        return addpattern(func, **kwargs)(base_func)
    return pattern_prepender
class _coconut_partial(_coconut_base_hashable):
    __slots__ = ("func", "_argdict", "_arglen", "_stargs", "keywords")
    if hasattr(_coconut.functools.partial, "__doc__"):
        __doc__ = _coconut.functools.partial.__doc__
    def __init__(self, _coconut_func, _coconut_argdict, _coconut_arglen, *args, **kwargs):
        self.func = _coconut_func
        self._argdict = _coconut_argdict
        self._arglen = _coconut_arglen
        self._stargs = args
        self.keywords = kwargs
    def __reduce__(self):
        return (self.__class__, (self.func, self._argdict, self._arglen) + self._stargs, self.keywords)
    def __setstate__(self, keywords):
        self.keywords = keywords
    @property
    def args(self):
        return _coconut.tuple(self._argdict.get(i) for i in _coconut.range(self._arglen)) + self._stargs
    def __call__(self, *args, **kwargs):
        callargs = []
        argind = 0
        for i in _coconut.range(self._arglen):
            if i in self._argdict:
                callargs.append(self._argdict[i])
            elif argind >= _coconut.len(args):
                raise _coconut.TypeError("expected at least " + _coconut.str(self._arglen - _coconut.len(self._argdict)) + " argument(s) to " + _coconut.repr(self))
            else:
                callargs.append(args[argind])
                argind += 1
        callargs += self._stargs
        callargs += args[argind:]
        kwargs.update(self.keywords)
        return self.func(*callargs, **kwargs)
    def __repr__(self):
        args = []
        for i in _coconut.range(self._arglen):
            if i in self._argdict:
                args.append(_coconut.repr(self._argdict[i]))
            else:
                args.append("?")
        for arg in self._stargs:
            args.append(_coconut.repr(arg))
        for k, v in self.keywords.items():
            args.append(k + "=" + _coconut.repr(v))
        return "%r$(%s)" % (self.func, ", ".join(args))
def consume(iterable, keep_last=0):
    """consume(iterable, keep_last) fully exhausts iterable and returns the last keep_last elements."""
    return _coconut.collections.deque(iterable, maxlen=keep_last)
class starmap(_coconut_base_hashable, _coconut.itertools.starmap):
    __slots__ = ("func", "iter")
    if hasattr(_coconut.itertools.starmap, "__doc__"):
        __doc__ = _coconut.itertools.starmap.__doc__
    def __new__(cls, function, iterable):
        new_map = _coconut.itertools.starmap.__new__(cls, function, iterable)
        new_map.func = function
        new_map.iter = iterable
        return new_map
    def __getitem__(self, index):
        if _coconut.isinstance(index, _coconut.slice):
            return self.__class__(self.func, _coconut_iter_getitem(self.iter, index))
        return self.func(*_coconut_iter_getitem(self.iter, index))
    def __reversed__(self):
        return self.__class__(self.func, *_coconut_reversed(self.iter))
    def __len__(self):
        return _coconut.len(self.iter)
    def __repr__(self):
        return "starmap(%r, %r)" % (self.func, self.iter)
    def __reduce__(self):
        return (self.__class__, (self.func, self.iter))
    def __iter__(self):
        return _coconut.iter(_coconut.itertools.starmap(self.func, self.iter))
    def __fmap__(self, func):
        return self.__class__(_coconut_forward_compose(self.func, func), self.iter)
def makedata(data_type, *args):
    """Construct an object of the given data_type containing the given arguments."""
    if _coconut.hasattr(data_type, "_make") and _coconut.issubclass(data_type, _coconut.tuple):
        return data_type._make(args)
    if _coconut.issubclass(data_type, (_coconut.range, _coconut.abc.Iterator)):
        return args
    if _coconut.issubclass(data_type, _coconut.str):
        return "".join(args)
    return data_type(args)
def datamaker(data_type):
    """DEPRECATED: use makedata instead."""
    return _coconut.functools.partial(makedata, data_type)
def fmap(func, obj):
    """fmap(func, obj) creates a copy of obj with func applied to its contents.

    Override by defining obj.__fmap__(func). For numpy arrays, uses np.vectorize.
    """
    obj_fmap = _coconut.getattr(obj, "__fmap__", None)
    if obj_fmap is not None:
        try:
            result = obj_fmap(func)
        except _coconut.NotImplementedError:
            pass
        else:
            if result is not _coconut.NotImplemented:
                return result
    if obj.__class__.__module__ in ('numpy', 'pandas'):
        return _coconut.numpy.vectorize(func)(obj)
    return _coconut_makedata(obj.__class__, *(_coconut_starmap(func, obj.items()) if _coconut.isinstance(obj, _coconut.abc.Mapping) else _coconut_map(func, obj)))
def memoize(maxsize=None, *args, **kwargs):
    """Decorator that memoizes a function, preventing it from being recomputed
    if it is called multiple times with the same arguments."""
    return _coconut.functools.lru_cache(maxsize, *args, **kwargs)
def _coconut_call_set_names(cls): pass
class override(_coconut_base_hashable):
    __slots__ = ("func",)
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return _coconut.types.MethodType(self.func, obj)
    def __set_name__(self, obj, name):
        if not _coconut.hasattr(_coconut.super(obj, obj), name):
            raise _coconut.RuntimeError(obj.__name__ + "." + name + " marked with @override but not overriding anything")
    def __reduce__(self):
        return (self.__class__, (self.func,))
def reveal_type(obj):
    """Special function to get MyPy to print the type of the given expression.
    At runtime, reveal_type is the identity function."""
    return obj
def reveal_locals():
    """Special function to get MyPy to print the type of the current locals.
    At runtime, reveal_locals always returns None."""
    pass
def _coconut_handle_cls_kwargs(**kwargs):
    """Some code taken from six under the terms of its MIT license."""
    metaclass = kwargs.pop("metaclass", None)
    if kwargs and metaclass is None:
        raise _coconut.TypeError("unexpected keyword argument(s) in class definition: %r" % (kwargs,))
    def coconut_handle_cls_kwargs_wrapper(cls):
        if metaclass is None:
            return cls
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get("__slots__")
        if slots is not None:
            if _coconut.isinstance(slots, _coconut.str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop("__dict__", None)
        orig_vars.pop("__weakref__", None)
        if _coconut.hasattr(cls, "__qualname__"):
            orig_vars["__qualname__"] = cls.__qualname__
        return metaclass(cls.__name__, cls.__bases__, orig_vars, **kwargs)
    return coconut_handle_cls_kwargs_wrapper
def _coconut_handle_cls_stargs(*args):
    temp_names = ["_coconut_base_cls_%s" % (i,) for i in _coconut.range(_coconut.len(args))]
    ns = _coconut.dict(_coconut.zip(temp_names, args))
    _coconut_exec("class _coconut_cls_stargs_base(" + ", ".join(temp_names) + "): pass", ns)
    return ns["_coconut_cls_stargs_base"]
def _coconut_dict_merge(*dicts, **kwargs):
    for_func = kwargs.pop("for_func", False)
    assert not kwargs, "error with internal Coconut function _coconut_dict_merge (you should report this at https://github.com/evhub/coconut/issues/new)"
    newdict = {}
    prevlen = 0
    for d in dicts:
        newdict.update(d)
        if for_func:
            if _coconut.len(newdict) != prevlen + _coconut.len(d):
                raise _coconut.TypeError("multiple values for the same keyword argument")
            prevlen = _coconut.len(newdict)
    return newdict
def ident(x, **kwargs):
    """The identity function. Generally equivalent to x -> x. Useful in point-free programming.
    Accepts one keyword-only argument, side_effect, which specifies a function to call on the argument before it is returned."""
    side_effect = kwargs.pop("side_effect", None)
    if kwargs:
        raise _coconut.TypeError("ident() got unexpected keyword arguments " + _coconut.repr(kwargs))
    if side_effect is not None:
        side_effect(x)
    return x
def of(_coconut_f, *args, **kwargs):
    """Function application operator function.

    Equivalent to:
        def of(f, *args, **kwargs) = f(*args, **kwargs).
    """
    return _coconut_f(*args, **kwargs)
class flip(_coconut_base_hashable):
    """Given a function, return a new function with inverse argument order.
    If nargs is passed, only the first nargs arguments are reversed."""
    __slots__ = ("func", "nargs")
    def __init__(self, func, nargs=None):
        self.func = func
        self.nargs = nargs
    def __reduce__(self):
        return (self.__class__, (self.func, self.nargs))
    def __call__(self, *args, **kwargs):
        return self.func(*args[::-1], **kwargs) if self.nargs is None else self.func(*(args[self.nargs-1::-1] + args[self.nargs:]), **kwargs)
    def __repr__(self):
        return "flip(%r%s)" % (self.func, "" if self.nargs is None else ", " + _coconut.repr(self.nargs))
class const(_coconut_base_hashable):
    """Create a function that, whatever its arguments, just returns the given value."""
    __slots__ = ("value",)
    def __init__(self, value):
        self.value = value
    def __reduce__(self):
        return (self.__class__, (self.value,))
    def __call__(self, *args, **kwargs):
        return self.value
    def __repr__(self):
        return "const(%s)" % (_coconut.repr(self.value),)
class _coconut_lifted(_coconut_base_hashable):
    __slots__ = ("func", "func_args", "func_kwargs")
    def __init__(self, _coconut_func, *func_args, **func_kwargs):
        self.func = _coconut_func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
    def __reduce__(self):
        return (self.__class__, (self.func,) + self.func_args, self.func_kwargs)
    def __setstate__(self, func_kwargs):
        self.func_kwargs = func_kwargs
    def __call__(self, *args, **kwargs):
        return self.func(*(g(*args, **kwargs) for g in self.func_args), **_coconut.dict((k, h(*args, **kwargs)) for k, h in self.func_kwargs.items()))
    def __repr__(self):
        return "lift(%r)(%s%s)" % (self.func, ", ".join(_coconut.repr(g) for g in self.func_args), ", ".join(k + "=" + _coconut.repr(h) for k, h in self.func_kwargs.items()))
class lift(_coconut_base_hashable):
    """Lifts a function up so that all of its arguments are functions.

    For a binary function f(x, y) and two unary functions g(z) and h(z), lift works as the S' combinator:
        lift(f)(g, h)(z) == f(g(z), h(z))

    In general, lift is requivalent to:
        def lift(f) = ((*func_args, **func_kwargs) -> (*args, **kwargs) ->
            f(*(g(*args, **kwargs) for g in func_args), **{k: h(*args, **kwargs) for k, h in func_kwargs.items()}))

    lift also supports a shortcut form such that lift(f, *func_args, **func_kwargs) is equivalent to lift(f)(*func_args, **func_kwargs).
    """
    __slots__ = ("func",)
    def __new__(cls, func, *func_args, **func_kwargs):
        self = _coconut.object.__new__(cls)
        self.func = func
        if func_args or func_kwargs:
            self = self(*func_args, **func_kwargs)
        return self
    def __reduce__(self):
        return (self.__class__, (self.func,))
    def __call__(self, *func_args, **func_kwargs):
        return _coconut_lifted(self.func, *func_args, **func_kwargs)
    def __repr__(self):
        return "lift(%r)" % (self.func,)
def all_equal(iterable):
    """For a given iterable, check whether all elements in that iterable are equal to each other.

    Assumes transitivity and 'x != y' being equivalent to 'not (x == y)'.
    """
    first_item = _coconut_sentinel
    for item in iterable:
        if first_item is _coconut_sentinel:
            first_item = item
        elif first_item != item:
            return False
    return True
def match_if(obj, predicate):
    """Meant to be used in infix pattern-matching expressions to match the left-hand side only if the predicate on the right-hand side holds.

    For example:
        a `match_if` predicate or b = obj

    The actual definition of match_if is extremely simple:
        def match_if(obj, predicate) = predicate(obj)
    """
    return predicate(obj)
def collectby(key_func, iterable, value_func=None, reduce_func=None):
    """Collect the items in iterable into a dictionary of lists keyed by key_func(item).

    if value_func is passed, collect value_func(item) into each list instead of item.

    If reduce_func is passed, instead of collecting the items into lists, reduce over
    the items of each key with reduce_func, effectively implementing a MapReduce operation.
    """
    collection = _coconut.collections.defaultdict(_coconut.list) if reduce_func is None else {}
    for item in iterable:
        key = key_func(item)
        if value_func is not None:
            item = value_func(item)
        if reduce_func is None:
            collection[key].append(item)
        else:
            old_item = collection.get(key, _coconut_sentinel)
            if old_item is not _coconut_sentinel:
                item = reduce_func(old_item, item)
            collection[key] = item
    return collection
def _namedtuple_of(**kwargs):
    """Construct an anonymous namedtuple of the given keyword arguments."""
    return _coconut.collections.namedtuple("_namedtuple_of", kwargs.keys())(*kwargs.values())
def _coconut_ndim(arr):
    if arr.__class__.__module__ in ('numpy', 'pandas') and _coconut.isinstance(arr, _coconut.numpy.ndarray):
        return arr.ndim
    if not _coconut.isinstance(arr, _coconut.abc.Sequence):
        return 0
    if _coconut.len(arr) == 0:
        return 1
    arr_dim = 1
    inner_arr = arr[0]
    while _coconut.isinstance(inner_arr, _coconut.abc.Sequence):
        arr_dim += 1
        if _coconut.len(inner_arr) < 1:
            break
        inner_arr = inner_arr[0]
    return arr_dim
def _coconut_expand_arr(arr, new_dims):
    if arr.__class__.__module__ in ('numpy', 'pandas') and _coconut.isinstance(arr, _coconut.numpy.ndarray):
        return arr.reshape((1,) * new_dims + arr.shape)
    for _ in _coconut.range(new_dims):
        arr = [arr]
    return arr
def _coconut_concatenate(arrs, axis):
    if _coconut.any(a.__class__.__module__ in ('numpy', 'pandas') for a in arrs):
        return _coconut.numpy.concatenate(arrs, axis)
    if not axis:
        return _coconut.list(_coconut.itertools.chain.from_iterable(arrs))
    return [_coconut_concatenate(rows, axis - 1) for rows in _coconut.zip(*arrs)]
def _coconut_multi_dim_arr(arrs, dim):
    arr_dims = [_coconut_ndim(a) for a in arrs]
    arrs = [_coconut_expand_arr(a, dim - d) if d < dim else a for a, d in _coconut.zip(arrs, arr_dims)]
    arr_dims.append(dim)
    max_arr_dim = _coconut.max(arr_dims)
    return _coconut_concatenate(arrs, max_arr_dim - dim)
_coconut_self_match_types = (bool, bytearray, bytes, dict, float, frozenset, int, list, set, str, tuple)
_coconut_MatchError, _coconut_count, _coconut_enumerate, _coconut_filter, _coconut_makedata, _coconut_map, _coconut_reiterable, _coconut_reversed, _coconut_starmap, _coconut_tee, _coconut_zip, TYPE_CHECKING, reduce, takewhile, dropwhile = MatchError, count, enumerate, filter, makedata, map, reiterable, reversed, starmap, tee, zip, False, _coconut.functools.reduce, _coconut.itertools.takewhile, _coconut.itertools.dropwhile

# Compiled Coconut: -----------------------------------------------------------

import coconut.convenience  #1 (line num in coconut source)

import os  #3 (line num in coconut source)
import time  #3 (line num in coconut source)
import datetime  #3 (line num in coconut source)
from itertools import repeat  #4 (line num in coconut source)
import torch as pt  #5 (line num in coconut source)
import torch_optimizer as optim  #6 (line num in coconut source)
from torch.utils.tensorboard import SummaryWriter  #7 (line num in coconut source)
import gym  #8 (line num in coconut source)
import gace  #8 (line num in coconut source)
import hace as ac  #9 (line num in coconut source)

## Defaults
algorithm: str = "td3"  # Name of used algorithm ∈  ./algorithm  #12 (line num in coconut source)
verbose: bool = True  # Print verbose debug output  #13 (line num in coconut source)
num_episodes: int = 42  # Number of episodes to play  #14 (line num in coconut source)
num_steps: int = 1  # How many steps to take  #15 (line num in coconut source)
num_iterations: int = 150  # Iterations per Epoch  #16 (line num in coconut source)
num_epochs: int = 100  # How many time steps to update policy  #17 (line num in coconut source)
early_stop: float = -500.0  # Early stop criterion  #18 (line num in coconut source)
batch_size: int = 100  # size of the batches during epoch  #19 (line num in coconut source)
rng_seed: int = 666  # Random seed for reproducability  #20 (line num in coconut source)

## GACE Settings
ace_id: str = "op2"  # ACE Identifier of the Environment  #23 (line num in coconut source)
ace_backend: str = "xh035"  # PDK/Technology backend of the ACE Environment  #24 (line num in coconut source)
ace_variant: int = 0  # ACE Environment variant  #25 (line num in coconut source)

## Hyper Parameters
update_interval: int = 2  # Update every n epochs  #28 (line num in coconut source)
γ: float = 0.99  # Discount Factor  #29 (line num in coconut source)
τ_soft: float = 1e-2  # Avantage Factor  #30 (line num in coconut source)
η: float = 1e-4  # Learning Rate  #31 (line num in coconut source)
β1: float = 0.9  #33 (line num in coconut source)
β2: float = 0.999  #34 (line num in coconut source)
βs: tuple[float] = (β1, β2)  # Noise added to replay actions  #35 (line num in coconut source)
σ_expl: float = 0.1  # Noise added to replay actions  #35 (line num in coconut source)
σ_smpl: float = 0.2  # Noise added to replay actions  #36 (line num in coconut source)
c: float = 0.5  # Noise clipping  #37 (line num in coconut source)
buffer_size: int = (int)(1e7)  # Maximum size of replay buffer  #38 (line num in coconut source)
warmup_periode: int = 50  # Collect experience during warmup  #39 (line num in coconut source)
d: int = 2  # Number of updates  #40 (line num in coconut source)

## Setup
env_id: str = f"gace:{ace_id}-{ace_backend}-v{ace_variant}"  #43 (line num in coconut source)
time_stamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  #44 (line num in coconut source)
model_dir: str = f"./models/{time_stamp}-{env_id}-{algorithm}"  #45 (line num in coconut source)
model_path: str = f"{model_dir}/checkpoint.pt"  #46 (line num in coconut source)
log_dir: str = f"./runs/{time_stamp}-{env_id}-{algorithm}/"  #47 (line num in coconut source)
os.makedirs(model_dir, exist_ok=True)  #48 (line num in coconut source)

## Setup Globals
device = pt.device("cuda:1") if pt.cuda.is_available() else pt.device("cpu")  #51 (line num in coconut source)
writer = SummaryWriter(log_dir=log_dir, flush_secs=30)  #52 (line num in coconut source)
_ = (pt.manual_seed)(rng_seed)  #53 (line num in coconut source)

## Utility
def write_performance(env: gym.Env, step: int):  #56 (line num in coconut source)
    target = env.target  #57 (line num in coconut source)
    performance = (ac.current_performance)((env).ace)  #58 (line num in coconut source)
    for k in target.keys():  #59 (line num in coconut source)
        writer.add_scalars(k, {"performance": performance[k], "target": target[k]}, step)  #60 (line num in coconut source)

## Data Processing
    return performance  #65 (line num in coconut source)

def process_gym(observations):  #65 (line num in coconut source)
    states = ((pt.vstack)([(pt.from_numpy)(obs) for obs in observations])).to(device)  #66 (line num in coconut source)

    return states  #69 (line num in coconut source)

def process_gace(observations, keys: dict[str, [list[str],]]):  #69 (line num in coconut source)
    fl = ["ugbw", "cof", "sr_f", "sr_r"]  #70 (line num in coconut source)
    ok = [k for k in keys['observations'] if (k[0].islower() or (k in keys['actions'])) and ("max-steps" not in k) and not k.startswith('vn_')]  #71 (line num in coconut source)
    idx = [keys['observations'].index(m) for m in ok]  #75 (line num in coconut source)
    idx_i = [ok.index(i) for i in ok if i.startswith('i') or i.endswith(':id')]  #76 (line num in coconut source)
    idx_v = [ok.index(v) for v in ok if v.startswith('voff_')]  #77 (line num in coconut source)
    idx_f = [ok.index(f) for f in ok if (not f.startswith('delta_') and ((any)((list)(filter(lambda f_: (f_ in f), fl))))) or f.endswith(':fug')]  #78 (line num in coconut source)
    msk_i = (_coconut_partial(pt.tensor, {}, 1, dtype=pt.bool))([i in idx_i for i in ((range)((len)(ok)))])  #82 (line num in coconut source)
    msk_v = (_coconut_partial(pt.tensor, {}, 1, dtype=pt.bool))([v in idx_v for v in ((range)((len)(ok)))])  #84 (line num in coconut source)
    msk_f = (_coconut_partial(pt.tensor, {}, 1, dtype=pt.bool))([f in idx_f for f in ((range)((len)(ok)))])  #86 (line num in coconut source)
    obs = ((pt.vstack)((list)((map)(pt.from_numpy, observations))))[_coconut.slice(None, None), idx]  #88 (line num in coconut source)
    obs_ = (_coconut_partial(pt.where, {0: msk_i, 1: obs * 1e6}, 3))((_coconut_partial(pt.where, {0: msk_f, 1: ((pt.log10)((pt.abs)(obs)))}, 3))(obs))  #89 (line num in coconut source)
#|> pt.where$(msk_v, obs * 1e6, ?)  \
    states = ((_coconut_partial(pt.nan_to_num, {}, 1, nan=0.0, posinf=0.0, neginf=0.0))(obs_)).to(device)  #92 (line num in coconut source)

    return states  #95 (line num in coconut source)

def scale_rewards(reward: pt.Tensor, ρ: float=1e-3):  #95 (line num in coconut source)
    scaled_reward = (reward - pt.mean(reward)) / (pt.std(reward) + ρ)  #96 (line num in coconut source)

## Replay Buffer
    return scaled_reward  #99 (line num in coconut source)

class ReplayBuffer(_coconut.typing.NamedTuple("ReplayBuffer", [("state", pt.Tensor), ("action", pt.Tensor), ("reward", pt.Tensor), ("next_state", pt.Tensor), ("done", pt.Tensor)])):  #99 (line num in coconut source)
    _coconut_is_data = True  #99 (line num in coconut source)
    __slots__ = ()  #99 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #99 (line num in coconut source)
    def __eq__(self, other):  #99 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #99 (line num in coconut source)
    def __hash__(self):  #99 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #99 (line num in coconut source)
    __match_args__ = ('state', 'action', 'reward', 'next_state', 'done')  #99 (line num in coconut source)
    @_coconut_tco  #99 (line num in coconut source)
    def __len__(self):  #99 (line num in coconut source)
        return _coconut_tail_call((int), ((self.state).shape)[0])  #101 (line num in coconut source)


def empty_buffer():  #103 (line num in coconut source)
    buffer = (ReplayBuffer)(*_coconut_iter_getitem((repeat)(pt.empty(0, device=device)), (_coconut.slice(None, 5))))  #104 (line num in coconut source)

    return buffer  #106 (line num in coconut source)

def push(buffer, other):  #106 (line num in coconut source)
    new = ((ReplayBuffer)(*(map)(_coconut_forward_compose(_coconut.operator.itemgetter((_coconut.slice(-buffer_size, None))), _coconut.operator.methodcaller("to", device)), other)) if len(buffer.state) <= 0 else (ReplayBuffer)(*(map)(_coconut_forward_compose(pt.vstack, _coconut.operator.itemgetter((_coconut.slice(-buffer_size, None))), _coconut.operator.methodcaller("to", device)), zip(buffer, other))))  #107 (line num in coconut source)

    return new  #114 (line num in coconut source)

def sample(buffer, batch_size: int):  #114 (line num in coconut source)
    idx = (((pt.randperm)((len)(buffer)))[_coconut.slice(None, batch_size)]).tolist()  #115 (line num in coconut source)
    smpl = (ReplayBuffer)(*(b[idx, :] for b in buffer))  #116 (line num in coconut source)

## Neural Networks
    return smpl  #119 (line num in coconut source)

class HStack(_coconut.collections.namedtuple("HStack", ()), pt.nn.Module):  #119 (line num in coconut source)
    _coconut_is_data = True  #119 (line num in coconut source)
    __slots__ = ()  #119 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #119 (line num in coconut source)
    def __eq__(self, other):  #119 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #119 (line num in coconut source)
    def __hash__(self):  #119 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #119 (line num in coconut source)
    __match_args__ = ()  #119 (line num in coconut source)
    @_coconut_tco  #119 (line num in coconut source)
    def forward(self, X: tuple[pt.Tensor, pt.Tensor]):  #119 (line num in coconut source)
        return _coconut_tail_call((pt.hstack), X)  #120 (line num in coconut source)


class VStack(_coconut.collections.namedtuple("VStack", ()), pt.nn.Module):  #122 (line num in coconut source)
    _coconut_is_data = True  #122 (line num in coconut source)
    __slots__ = ()  #122 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #122 (line num in coconut source)
    def __eq__(self, other):  #122 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #122 (line num in coconut source)
    def __hash__(self):  #122 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #122 (line num in coconut source)
    __match_args__ = ()  #122 (line num in coconut source)
    @_coconut_tco  #122 (line num in coconut source)
    def forward(self, X: tuple[pt.Tensor, pt.Tensor]):  #122 (line num in coconut source)
        return _coconut_tail_call((pt.vstack), X)  #123 (line num in coconut source)

## Critic

def critic_net(obs_dim: int, act_dim: int):  #126 (line num in coconut source)
    dim = obs_dim + act_dim  #127 (line num in coconut source)
    crt_net = pt.nn.Sequential(HStack(), pt.nn.Linear(dim, 400), pt.nn.ReLU(), pt.nn.Linear(400, 300), pt.nn.ReLU(), pt.nn.Linear(300, 1))  #128 (line num in coconut source)

## Actor
    return crt_net  #134 (line num in coconut source)

def actor_net(obs_dim: int, act_dim: int):  #134 (line num in coconut source)
    act_net = pt.nn.Sequential(pt.nn.Linear(obs_dim, 400), pt.nn.ReLU(), pt.nn.Linear(400, 300), pt.nn.ReLU(), pt.nn.Linear(300, act_dim), pt.nn.Tanh())  #135 (line num in coconut source)

## TD3 Agent
    return act_net  #140 (line num in coconut source)

class Agent(_coconut.typing.NamedTuple("Agent", [("Q1", pt.nn.Module), ("Q2", pt.nn.Module), ("π", pt.nn.Module), ("θ1", pt.nn.Module), ("θ2", pt.nn.Module), ("φ", pt.nn.Module), ("Q1_optim", pt.optim.Optimizer), ("Q2_optim", pt.optim.Optimizer), ("π_optim", pt.optim.Optimizer), ("Q_loss", _coconut.typing.Any)])):  #140 (line num in coconut source)
    _coconut_is_data = True  #140 (line num in coconut source)
    __slots__ = ()  #140 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #140 (line num in coconut source)
    def __eq__(self, other):  #140 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #140 (line num in coconut source)
    def __hash__(self):  #140 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #140 (line num in coconut source)
    __match_args__ = ('Q1', 'Q2', 'π', 'θ1', 'θ2', 'φ', 'Q1_optim', 'Q2_optim', 'π_optim', 'Q_loss')  #140 (line num in coconut source)
    def save_state(self, checkpoint_file: str):  #140 (line num in coconut source)
        state_dicts = (fmap)(_coconut.operator.methodcaller("state_dict"), [self.Q1, self.Q2, self.π, self.θ1, self.θ2, self.φ, self.Q1_optim, self.Q2_optim, self.π_optim])  #145 (line num in coconut source)
        keys = ["Q1", "Q2", "pi", "T1", "T2", "phi", "Q1_opt", "Q2_opt", "pi_opt"]  #148 (line num in coconut source)
        save_dict = (dict)((zip)(*(keys, state_dicts)))  #150 (line num in coconut source)
        res = pt.save(save_dict, checkpoint_file)  #151 (line num in coconut source)
        return res  #152 (line num in coconut source)

    def load_state(self):  #152 (line num in coconut source)
        raise (NotImplementedError)  #153 (line num in coconut source)


def make_agent(act_dim: int, obs_dim: int):  #155 (line num in coconut source)
    Q1 = (critic_net(obs_dim, act_dim)).to(device)  #156 (line num in coconut source)
    Q2 = (critic_net(obs_dim, act_dim)).to(device)  #157 (line num in coconut source)
    π = (actor_net(obs_dim, act_dim)).to(device)  #158 (line num in coconut source)
    θ1 = (critic_net(obs_dim, act_dim)).to(device)  #159 (line num in coconut source)
    θ2 = (critic_net(obs_dim, act_dim)).to(device)  #160 (line num in coconut source)
    φ = (actor_net(obs_dim, act_dim)).to(device)  #161 (line num in coconut source)
    _ = soft_update(Q1, θ1, τ=1.0)  #162 (line num in coconut source)
    _ = soft_update(Q2, θ2, τ=1.0)  #163 (line num in coconut source)
    _ = soft_update(π, φ, τ=1.0)  #164 (line num in coconut source)
    Q1_optim = pt.optim.Adam(Q1.parameters(), lr=η, betas=βs)  #165 (line num in coconut source)
    Q2_optim = pt.optim.Adam(Q2.parameters(), lr=η, betas=βs)  #166 (line num in coconut source)
    π_optim = pt.optim.Adam(π.parameters(), lr=η, betas=βs)  #167 (line num in coconut source)
    Q_loss = pt.nn.functional.mse_loss  #168 (line num in coconut source)
    agent = Agent(Q1, Q2, π, θ1, θ2, φ, Q1_optim, Q2_optim, π_optim, Q_loss)  #169 (line num in coconut source)

    return agent  #171 (line num in coconut source)

def soft_update(source: pt.nn.Module, target: pt.nn.Module, τ: float=1e-2):  #171 (line num in coconut source)
    for param, target_param in zip(source.parameters(), target.parameters()):  #173 (line num in coconut source)
        target_param.data.copy_(τ * param.data + (1 - τ) * target_param.data)  #174 (line num in coconut source)

    return target  #176 (line num in coconut source)

def random_action(envs):  #176 (line num in coconut source)
    action = (((pt.vstack)((list)((map)(_coconut_forward_compose(_coconut.operator.methodcaller("sample"), pt.from_numpy), envs.action_space)))).detach()).to(device)  #177 (line num in coconut source)

## Update Policy
    return action  #181 (line num in coconut source)

def update_actor(iteration: int, epoch: int, agent: Agent, states: pt.Tensor):  #181 (line num in coconut source)
    π_loss = ((_coconut_minus))((pt.mean)((agent.Q1)((_coconut_partial((_coconut_comma_op), {0: states}, 2))((agent.π)(states)))))  #183 (line num in coconut source)
    _ = agent.π_optim.zero_grad()  #184 (line num in coconut source)
    _ = π_loss.backward()  #185 (line num in coconut source)
    _ = agent.π_optim.step()  #186 (line num in coconut source)
    _ = soft_update(agent.Q1, agent.θ1, τ=τ_soft)  #187 (line num in coconut source)
    _ = soft_update(agent.Q2, agent.θ2, τ=τ_soft)  #188 (line num in coconut source)
    _ = soft_update(agent.π, agent.φ, τ=τ_soft)  #189 (line num in coconut source)
    _ = writer.add_scalar("_Loss_π", π_loss, epoch)  #190 (line num in coconut source)

    return π_loss  #192 (line num in coconut source)

@_coconut_mark_as_match  #192 (line num in coconut source)
def update_step(*_coconut_match_args, **_coconut_match_kwargs):  #192 (line num in coconut source)
    _coconut_match_check_0 = False  #192 (line num in coconut source)
    _coconut_match_set_name_iteration = _coconut_sentinel  #192 (line num in coconut source)
    _coconut_match_set_name_agent = _coconut_sentinel  #192 (line num in coconut source)
    _coconut_match_set_name_losses = _coconut_sentinel  #192 (line num in coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #192 (line num in coconut source)
    if (4 <= _coconut.len(_coconut_match_args) <= 5) and ("iteration" not in _coconut_match_kwargs) and ("agent" not in _coconut_match_kwargs) and (_coconut.sum((_coconut.len(_coconut_match_args) > 4, "losses" in _coconut_match_kwargs)) == 1) and (_coconut_match_args[1] == 0):  #192 (line num in coconut source)
        _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("iteration")  #192 (line num in coconut source)
        _coconut_match_temp_1 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("agent")  #192 (line num in coconut source)
        _coconut_match_temp_2 = _coconut_match_args[4] if _coconut.len(_coconut_match_args) > 4 else _coconut_match_kwargs.pop("losses")  #192 (line num in coconut source)
        if not _coconut_match_kwargs:  #192 (line num in coconut source)
            _coconut_match_set_name_iteration = _coconut_match_temp_0  #192 (line num in coconut source)
            _coconut_match_set_name_agent = _coconut_match_temp_1  #192 (line num in coconut source)
            _coconut_match_set_name_losses = _coconut_match_temp_2  #192 (line num in coconut source)
            _coconut_match_check_0 = True  #192 (line num in coconut source)
    if _coconut_match_check_0:  #192 (line num in coconut source)
        if _coconut_match_set_name_iteration is not _coconut_sentinel:  #192 (line num in coconut source)
            iteration = _coconut_match_temp_0  #192 (line num in coconut source)
        if _coconut_match_set_name_agent is not _coconut_sentinel:  #192 (line num in coconut source)
            agent = _coconut_match_temp_1  #192 (line num in coconut source)
        if _coconut_match_set_name_losses is not _coconut_sentinel:  #192 (line num in coconut source)
            losses = _coconut_match_temp_2  #192 (line num in coconut source)
    if not _coconut_match_check_0:  #192 (line num in coconut source)
        raise _coconut_FunctionMatchError('def update_step( iteration, 0, agent, _, losses\n) =  (q1_loss, q2_loss, \u03c0_loss) where:', _coconut_match_args)  #192 (line num in coconut source)

    (q1_loss, q2_loss, π_loss_) = losses  #194 (line num in coconut source)
    π_loss = π_loss_[~π_loss_.isnan()]  #195 (line num in coconut source)
    return (q1_loss, q2_loss, π_loss)  #196 (line num in coconut source)

@_coconut_addpattern(update_step)  #196 (line num in coconut source)
@_coconut_tco  #196 (line num in coconut source)
@_coconut_mark_as_match  #196 (line num in coconut source)
def update_step(*_coconut_match_args, **_coconut_match_kwargs):  #196 (line num in coconut source)
    _coconut_match_check_1 = False  #196 (line num in coconut source)
    _coconut_match_set_name_iteration = _coconut_sentinel  #196 (line num in coconut source)
    _coconut_match_set_name_epoch = _coconut_sentinel  #196 (line num in coconut source)
    _coconut_match_set_name_agent = _coconut_sentinel  #196 (line num in coconut source)
    _coconut_match_set_name_smpl = _coconut_sentinel  #196 (line num in coconut source)
    _coconut_match_set_name_losses = _coconut_sentinel  #196 (line num in coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #196 (line num in coconut source)
    if (_coconut.len(_coconut_match_args) <= 5) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "iteration" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "epoch" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "agent" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 3, "smpl" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 4, "losses" in _coconut_match_kwargs)) == 1):  #196 (line num in coconut source)
        _coconut_match_temp_3 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("iteration")  #196 (line num in coconut source)
        _coconut_match_temp_4 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("epoch")  #196 (line num in coconut source)
        _coconut_match_temp_5 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("agent")  #196 (line num in coconut source)
        _coconut_match_temp_6 = _coconut_match_args[3] if _coconut.len(_coconut_match_args) > 3 else _coconut_match_kwargs.pop("smpl")  #196 (line num in coconut source)
        _coconut_match_temp_7 = _coconut_match_args[4] if _coconut.len(_coconut_match_args) > 4 else _coconut_match_kwargs.pop("losses")  #196 (line num in coconut source)
        if not _coconut_match_kwargs:  #196 (line num in coconut source)
            _coconut_match_set_name_iteration = _coconut_match_temp_3  #196 (line num in coconut source)
            _coconut_match_set_name_epoch = _coconut_match_temp_4  #196 (line num in coconut source)
            _coconut_match_set_name_agent = _coconut_match_temp_5  #196 (line num in coconut source)
            _coconut_match_set_name_smpl = _coconut_match_temp_6  #196 (line num in coconut source)
            _coconut_match_set_name_losses = _coconut_match_temp_7  #196 (line num in coconut source)
            _coconut_match_check_1 = True  #196 (line num in coconut source)
    if _coconut_match_check_1:  #196 (line num in coconut source)
        if _coconut_match_set_name_iteration is not _coconut_sentinel:  #196 (line num in coconut source)
            iteration = _coconut_match_temp_3  #196 (line num in coconut source)
        if _coconut_match_set_name_epoch is not _coconut_sentinel:  #196 (line num in coconut source)
            epoch = _coconut_match_temp_4  #196 (line num in coconut source)
        if _coconut_match_set_name_agent is not _coconut_sentinel:  #196 (line num in coconut source)
            agent = _coconut_match_temp_5  #196 (line num in coconut source)
        if _coconut_match_set_name_smpl is not _coconut_sentinel:  #196 (line num in coconut source)
            smpl = _coconut_match_temp_6  #196 (line num in coconut source)
        if _coconut_match_set_name_losses is not _coconut_sentinel:  #196 (line num in coconut source)
            losses = _coconut_match_temp_7  #196 (line num in coconut source)
    if not _coconut_match_check_1:  #196 (line num in coconut source)
        raise _coconut_FunctionMatchError('addpattern def update_step( iteration, epoch, agent, smpl, losses\n) = update_step( iteration, epoch_, agent\n, smpl, losses_ ) where:', _coconut_match_args)  #196 (line num in coconut source)

    states, actions, rewards, states_, dones = smpl  #199 (line num in coconut source)
    ε = ((_coconut_partial(pt.clamp, {1: -c, 2: c}, 3))((_coconut_partial(pt.normal, {1: σ_smpl}, 2))((pt.zeros_like)(actions)))).to(device)  #200 (line num in coconut source)
    actions_ = (_coconut_partial(pt.clamp, {1: -1.0, 2: 1.0}, 3))((agent.φ(states_) + ε))  #202 (line num in coconut source)
    q1_target = (agent.θ1)((states_, actions_))  #203 (line num in coconut source)
    q2_target = (agent.θ2)((states_, actions_))  #204 (line num in coconut source)
    q_target = pt.min(q1_target, q2_target)  #205 (line num in coconut source)
    y = ((rewards + (1.0 - dones) * γ * q_target)).detach()  #206 (line num in coconut source)
    q1 = (agent.Q1)((states, actions))  #207 (line num in coconut source)
    q2 = (agent.Q2)((states, actions))  #208 (line num in coconut source)
    q1_loss = agent.Q_loss(q1, y)  #209 (line num in coconut source)
    q2_loss = agent.Q_loss(q2, y)  #210 (line num in coconut source)
    _ = agent.Q1_optim.zero_grad()  #211 (line num in coconut source)
    _ = q1_loss.backward()  #212 (line num in coconut source)
    _ = agent.Q1_optim.step()  #213 (line num in coconut source)
    _ = agent.Q2_optim.zero_grad()  #214 (line num in coconut source)
    _ = q2_loss.backward()  #215 (line num in coconut source)
    _ = agent.Q2_optim.step()  #216 (line num in coconut source)
    π_loss = (update_actor(iteration, epoch, agent, states) if iteration in count(0, d) else ((pt.tensor)(pt.nan)).to(device))  #217 (line num in coconut source)
    _ = writer.add_scalar("_Loss_Q1", q1_loss, iteration)  #220 (line num in coconut source)
    _ = writer.add_scalar("_Loss_Q2", q2_loss, iteration)  #221 (line num in coconut source)
#if verbose:
#    print(f"Epoch {epoch:03} | Q Loss: {q1_loss:3f}, {q2_loss:3f}")
    epoch_ = epoch - 1  #224 (line num in coconut source)
    losses_ = (tuple)((map)(pt.cat, (_coconut_partial(zip, {0: losses}, 2))((map)(_coconut_forward_compose(_coconut.operator.methodcaller("detach"), _coconut.operator.itemgetter((None))), (q1_loss, q2_loss, π_loss)))))  #225 (line num in coconut source)

    return _coconut_tail_call(update_step, iteration, epoch_, agent, smpl, losses_)  #228 (line num in coconut source)

def update_policy(iteration: int, epochs: int, agent: Agent, buffer: ReplayBuffer):  #228 (line num in coconut source)
    smpl = sample(buffer, batch_size)  #230 (line num in coconut source)
    losses = (tuple)(_coconut_iter_getitem((repeat)(pt.empty(0, device=device)), (_coconut.slice(None, 3))))  #231 (line num in coconut source)
    losses_ = update_step(iteration, epochs, agent, smpl, losses)  #232 (line num in coconut source)

## Evaluate Policy
    return losses  #235 (line num in coconut source)

@_coconut_mark_as_match  #235 (line num in coconut source)
def evaluate_policy(*_coconut_match_args, **_coconut_match_kwargs):  #235 (line num in coconut source)
    _coconut_match_check_2 = False  #235 (line num in coconut source)
    _coconut_match_set_name_iteration = _coconut_sentinel  #235 (line num in coconut source)
    _coconut_match_set_name_agent = _coconut_sentinel  #235 (line num in coconut source)
    _coconut_match_set_name_envs = _coconut_sentinel  #235 (line num in coconut source)
    _coconut_match_set_name_states = _coconut_sentinel  #235 (line num in coconut source)
    _coconut_match_set_name_buffer = _coconut_sentinel  #235 (line num in coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #235 (line num in coconut source)
    if (2 <= _coconut.len(_coconut_match_args) <= 6) and ("iteration" not in _coconut_match_kwargs) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "agent" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 3, "envs" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 4, "states" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 5, "buffer" in _coconut_match_kwargs)) == 1) and (_coconut_match_args[1] == 0):  #235 (line num in coconut source)
        _coconut_match_temp_8 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("iteration")  #235 (line num in coconut source)
        _coconut_match_temp_9 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("agent")  #235 (line num in coconut source)
        _coconut_match_temp_10 = _coconut_match_args[3] if _coconut.len(_coconut_match_args) > 3 else _coconut_match_kwargs.pop("envs")  #235 (line num in coconut source)
        _coconut_match_temp_11 = _coconut_match_args[4] if _coconut.len(_coconut_match_args) > 4 else _coconut_match_kwargs.pop("states")  #235 (line num in coconut source)
        _coconut_match_temp_12 = _coconut_match_args[5] if _coconut.len(_coconut_match_args) > 5 else _coconut_match_kwargs.pop("buffer")  #235 (line num in coconut source)
        if not _coconut_match_kwargs:  #235 (line num in coconut source)
            _coconut_match_set_name_iteration = _coconut_match_temp_8  #235 (line num in coconut source)
            _coconut_match_set_name_agent = _coconut_match_temp_9  #235 (line num in coconut source)
            _coconut_match_set_name_envs = _coconut_match_temp_10  #235 (line num in coconut source)
            _coconut_match_set_name_states = _coconut_match_temp_11  #235 (line num in coconut source)
            _coconut_match_set_name_buffer = _coconut_match_temp_12  #235 (line num in coconut source)
            _coconut_match_check_2 = True  #235 (line num in coconut source)
    if _coconut_match_check_2:  #235 (line num in coconut source)
        if _coconut_match_set_name_iteration is not _coconut_sentinel:  #235 (line num in coconut source)
            iteration = _coconut_match_temp_8  #235 (line num in coconut source)
        if _coconut_match_set_name_agent is not _coconut_sentinel:  #235 (line num in coconut source)
            agent = _coconut_match_temp_9  #235 (line num in coconut source)
        if _coconut_match_set_name_envs is not _coconut_sentinel:  #235 (line num in coconut source)
            envs = _coconut_match_temp_10  #235 (line num in coconut source)
        if _coconut_match_set_name_states is not _coconut_sentinel:  #235 (line num in coconut source)
            states = _coconut_match_temp_11  #235 (line num in coconut source)
        if _coconut_match_set_name_buffer is not _coconut_sentinel:  #235 (line num in coconut source)
            buffer = _coconut_match_temp_12  #235 (line num in coconut source)
    if not _coconut_match_check_2:  #235 (line num in coconut source)
        raise _coconut_FunctionMatchError('def evaluate_policy( iteration, 0, agent, envs, states, buffer) = (buffer,states)', _coconut_match_args)  #235 (line num in coconut source)

    return (buffer, states)  #235 (line num in coconut source)

@_coconut_addpattern(evaluate_policy)  #236 (line num in coconut source)
@_coconut_tco  #236 (line num in coconut source)
@_coconut_mark_as_match  #236 (line num in coconut source)
def evaluate_policy(*_coconut_match_args, **_coconut_match_kwargs):  #236 (line num in coconut source)
    _coconut_match_check_3 = False  #236 (line num in coconut source)
    _coconut_match_set_name_iteration = _coconut_sentinel  #236 (line num in coconut source)
    _coconut_match_set_name_step = _coconut_sentinel  #236 (line num in coconut source)
    _coconut_match_set_name_agent = _coconut_sentinel  #236 (line num in coconut source)
    _coconut_match_set_name_envs = _coconut_sentinel  #236 (line num in coconut source)
    _coconut_match_set_name_states = _coconut_sentinel  #236 (line num in coconut source)
    _coconut_match_set_name_buffer = _coconut_sentinel  #236 (line num in coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #236 (line num in coconut source)
    if (_coconut.len(_coconut_match_args) <= 6) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "iteration" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "step" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "agent" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 3, "envs" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 4, "states" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 5, "buffer" in _coconut_match_kwargs)) == 1):  #236 (line num in coconut source)
        _coconut_match_temp_13 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("iteration")  #236 (line num in coconut source)
        _coconut_match_temp_14 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("step")  #236 (line num in coconut source)
        _coconut_match_temp_15 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("agent")  #236 (line num in coconut source)
        _coconut_match_temp_16 = _coconut_match_args[3] if _coconut.len(_coconut_match_args) > 3 else _coconut_match_kwargs.pop("envs")  #236 (line num in coconut source)
        _coconut_match_temp_17 = _coconut_match_args[4] if _coconut.len(_coconut_match_args) > 4 else _coconut_match_kwargs.pop("states")  #236 (line num in coconut source)
        _coconut_match_temp_18 = _coconut_match_args[5] if _coconut.len(_coconut_match_args) > 5 else _coconut_match_kwargs.pop("buffer")  #236 (line num in coconut source)
        if not _coconut_match_kwargs:  #236 (line num in coconut source)
            _coconut_match_set_name_iteration = _coconut_match_temp_13  #236 (line num in coconut source)
            _coconut_match_set_name_step = _coconut_match_temp_14  #236 (line num in coconut source)
            _coconut_match_set_name_agent = _coconut_match_temp_15  #236 (line num in coconut source)
            _coconut_match_set_name_envs = _coconut_match_temp_16  #236 (line num in coconut source)
            _coconut_match_set_name_states = _coconut_match_temp_17  #236 (line num in coconut source)
            _coconut_match_set_name_buffer = _coconut_match_temp_18  #236 (line num in coconut source)
            _coconut_match_check_3 = True  #236 (line num in coconut source)
    if _coconut_match_check_3:  #236 (line num in coconut source)
        if _coconut_match_set_name_iteration is not _coconut_sentinel:  #236 (line num in coconut source)
            iteration = _coconut_match_temp_13  #236 (line num in coconut source)
        if _coconut_match_set_name_step is not _coconut_sentinel:  #236 (line num in coconut source)
            step = _coconut_match_temp_14  #236 (line num in coconut source)
        if _coconut_match_set_name_agent is not _coconut_sentinel:  #236 (line num in coconut source)
            agent = _coconut_match_temp_15  #236 (line num in coconut source)
        if _coconut_match_set_name_envs is not _coconut_sentinel:  #236 (line num in coconut source)
            envs = _coconut_match_temp_16  #236 (line num in coconut source)
        if _coconut_match_set_name_states is not _coconut_sentinel:  #236 (line num in coconut source)
            states = _coconut_match_temp_17  #236 (line num in coconut source)
        if _coconut_match_set_name_buffer is not _coconut_sentinel:  #236 (line num in coconut source)
            buffer = _coconut_match_temp_18  #236 (line num in coconut source)
    if not _coconut_match_check_3:  #236 (line num in coconut source)
        raise _coconut_FunctionMatchError('addpattern def evaluate_policy( iteration, step, agent, envs, states, buffer\n    ) = evaluate_policy( iteration, step_, agent\n    , envs, states_, buffer_\n    ) where:', _coconut_match_args)  #236 (line num in coconut source)

    ε = (pt.distributions.Normal(0, σ_expl)).sample()  #240 (line num in coconut source)
    with pt.no_grad():  #241 (line num in coconut source)
        actions = (random_action(envs) if (iteration * len(envs)) < warmup_periode else (_coconut_partial(pt.clamp, {1: -1.0, 2: 1.0}, 3))((agent.π(states) + ε).detach()))  #242 (line num in coconut source)
    t0 = time.time()  #246 (line num in coconut source)
    observations_, rewards_, dones_, infos = (envs.step)((list)((fmap)(_coconut_forward_compose(_coconut.operator.methodcaller("squeeze"), _coconut.operator.methodcaller("cpu"), _coconut.operator.methodcaller("numpy")), (_coconut_partial(pt.split, {1: 1}, 2))(actions))))  #247 (line num in coconut source)
    t1 = time.time()  #251 (line num in coconut source)
    keys = infos[0]  #252 (line num in coconut source)
    observations = envs.reset(done_mask=dones_)  #253 (line num in coconut source)
    states_ = process_gace(observations, keys)  #254 (line num in coconut source)
    rewards = (((pt.tensor)(rewards_)).to(device)).reshape(-1, 1)  #255 (line num in coconut source)
    dones = ((pt.tensor(dones_, device=device, dtype=pt.int))).reshape(-1, 1)  #256 (line num in coconut source)
    buffer_ = (push)(buffer, (states, actions, rewards, states_, dones))  #258 (line num in coconut source)
    _ = writer.add_scalar("_Reward_Mean", rewards.mean().item(), step)  #259 (line num in coconut source)
    if verbose:  #260 (line num in coconut source)
        dt = (t1 - t0)  #261 (line num in coconut source)
        print(f"{iteration:03}/{step:03} took {dt:.3f}s | Average Reward: {rewards.mean():.3f}")  #262 (line num in coconut source)
        if any(dones_):  #263 (line num in coconut source)
            de = [i for i, d in enumerate(dones_) if d]  #264 (line num in coconut source)
            print(f"\tEnvironments {de} are done in Iteration {iteration:03}.")  #265 (line num in coconut source)
    step_ = step - 1  #266 (line num in coconut source)

## Run Episode until done
    return _coconut_tail_call(evaluate_policy, iteration, step_, agent, envs, states_, buffer_)  #269 (line num in coconut source)

@_coconut_mark_as_match  #269 (line num in coconut source)
def run_algorithm(*_coconut_match_args, **_coconut_match_kwargs):  #269 (line num in coconut source)
    _coconut_match_check_4 = False  #269 (line num in coconut source)
    _coconut_match_set_name_agent = _coconut_sentinel  #269 (line num in coconut source)
    _coconut_match_set_name_episode = _coconut_sentinel  #269 (line num in coconut source)
    _coconut_match_set_name_buffer = _coconut_sentinel  #269 (line num in coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #269 (line num in coconut source)
    if (_coconut.len(_coconut_match_args) == 7) and ("agent" not in _coconut_match_kwargs) and ("episode" not in _coconut_match_kwargs) and ("buffer" not in _coconut_match_kwargs) and (_coconut_match_args[6] is True):  #269 (line num in coconut source)
        _coconut_match_temp_19 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("agent")  #269 (line num in coconut source)
        _coconut_match_temp_20 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("episode")  #269 (line num in coconut source)
        _coconut_match_temp_21 = _coconut_match_args[5] if _coconut.len(_coconut_match_args) > 5 else _coconut_match_kwargs.pop("buffer")  #269 (line num in coconut source)
        if not _coconut_match_kwargs:  #269 (line num in coconut source)
            _coconut_match_set_name_agent = _coconut_match_temp_19  #269 (line num in coconut source)
            _coconut_match_set_name_episode = _coconut_match_temp_20  #269 (line num in coconut source)
            _coconut_match_set_name_buffer = _coconut_match_temp_21  #269 (line num in coconut source)
            _coconut_match_check_4 = True  #269 (line num in coconut source)
    if _coconut_match_check_4:  #269 (line num in coconut source)
        if _coconut_match_set_name_agent is not _coconut_sentinel:  #269 (line num in coconut source)
            agent = _coconut_match_temp_19  #269 (line num in coconut source)
        if _coconut_match_set_name_episode is not _coconut_sentinel:  #269 (line num in coconut source)
            episode = _coconut_match_temp_20  #269 (line num in coconut source)
        if _coconut_match_set_name_buffer is not _coconut_sentinel:  #269 (line num in coconut source)
            buffer = _coconut_match_temp_21  #269 (line num in coconut source)
    if not _coconut_match_check_4:  #269 (line num in coconut source)
        raise _coconut_FunctionMatchError('def run_algorithm(_, agent, episode, _, _, buffer, True) = agent where:', _coconut_match_args)  #269 (line num in coconut source)

    total = ((pt.sum)(buffer.reward)).item()  #270 (line num in coconut source)
    _ = writer.add_scalar(f"_Reward_Total", total, episode)  #271 (line num in coconut source)
    if verbose:  #272 (line num in coconut source)
        (print)(f"Episode {episode:03} Finished | Total Reward: {total}")  #273 (line num in coconut source)
    return agent  #274 (line num in coconut source)

@_coconut_addpattern(run_algorithm)  #274 (line num in coconut source)
@_coconut_tco  #274 (line num in coconut source)
@_coconut_mark_as_match  #274 (line num in coconut source)
def run_algorithm(*_coconut_match_args, **_coconut_match_kwargs):  #274 (line num in coconut source)
    _coconut_match_check_5 = False  #274 (line num in coconut source)
    _coconut_match_set_name_envs = _coconut_sentinel  #274 (line num in coconut source)
    _coconut_match_set_name_agent = _coconut_sentinel  #274 (line num in coconut source)
    _coconut_match_set_name_episode = _coconut_sentinel  #274 (line num in coconut source)
    _coconut_match_set_name_iteration = _coconut_sentinel  #274 (line num in coconut source)
    _coconut_match_set_name_states = _coconut_sentinel  #274 (line num in coconut source)
    _coconut_match_set_name_buffer = _coconut_sentinel  #274 (line num in coconut source)
    _coconut_match_set_name_done = _coconut_sentinel  #274 (line num in coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #274 (line num in coconut source)
    if (_coconut.len(_coconut_match_args) <= 7) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "envs" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "agent" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "episode" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 3, "iteration" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 4, "states" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 5, "buffer" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 6, "done" in _coconut_match_kwargs)) == 1):  #274 (line num in coconut source)
        _coconut_match_temp_22 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("envs")  #274 (line num in coconut source)
        _coconut_match_temp_23 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("agent")  #274 (line num in coconut source)
        _coconut_match_temp_24 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("episode")  #274 (line num in coconut source)
        _coconut_match_temp_25 = _coconut_match_args[3] if _coconut.len(_coconut_match_args) > 3 else _coconut_match_kwargs.pop("iteration")  #274 (line num in coconut source)
        _coconut_match_temp_26 = _coconut_match_args[4] if _coconut.len(_coconut_match_args) > 4 else _coconut_match_kwargs.pop("states")  #274 (line num in coconut source)
        _coconut_match_temp_27 = _coconut_match_args[5] if _coconut.len(_coconut_match_args) > 5 else _coconut_match_kwargs.pop("buffer")  #274 (line num in coconut source)
        _coconut_match_temp_28 = _coconut_match_args[6] if _coconut.len(_coconut_match_args) > 6 else _coconut_match_kwargs.pop("done")  #274 (line num in coconut source)
        if not _coconut_match_kwargs:  #274 (line num in coconut source)
            _coconut_match_set_name_envs = _coconut_match_temp_22  #274 (line num in coconut source)
            _coconut_match_set_name_agent = _coconut_match_temp_23  #274 (line num in coconut source)
            _coconut_match_set_name_episode = _coconut_match_temp_24  #274 (line num in coconut source)
            _coconut_match_set_name_iteration = _coconut_match_temp_25  #274 (line num in coconut source)
            _coconut_match_set_name_states = _coconut_match_temp_26  #274 (line num in coconut source)
            _coconut_match_set_name_buffer = _coconut_match_temp_27  #274 (line num in coconut source)
            _coconut_match_set_name_done = _coconut_match_temp_28  #274 (line num in coconut source)
            _coconut_match_check_5 = True  #274 (line num in coconut source)
    if _coconut_match_check_5:  #274 (line num in coconut source)
        if _coconut_match_set_name_envs is not _coconut_sentinel:  #274 (line num in coconut source)
            envs = _coconut_match_temp_22  #274 (line num in coconut source)
        if _coconut_match_set_name_agent is not _coconut_sentinel:  #274 (line num in coconut source)
            agent = _coconut_match_temp_23  #274 (line num in coconut source)
        if _coconut_match_set_name_episode is not _coconut_sentinel:  #274 (line num in coconut source)
            episode = _coconut_match_temp_24  #274 (line num in coconut source)
        if _coconut_match_set_name_iteration is not _coconut_sentinel:  #274 (line num in coconut source)
            iteration = _coconut_match_temp_25  #274 (line num in coconut source)
        if _coconut_match_set_name_states is not _coconut_sentinel:  #274 (line num in coconut source)
            states = _coconut_match_temp_26  #274 (line num in coconut source)
        if _coconut_match_set_name_buffer is not _coconut_sentinel:  #274 (line num in coconut source)
            buffer = _coconut_match_temp_27  #274 (line num in coconut source)
        if _coconut_match_set_name_done is not _coconut_sentinel:  #274 (line num in coconut source)
            done = _coconut_match_temp_28  #274 (line num in coconut source)
    if not _coconut_match_check_5:  #274 (line num in coconut source)
        raise _coconut_FunctionMatchError('addpattern def run_algorithm( envs, agent, episode, iteration, states, buffer, done\n) = run_algorithm( envs, agent, episode, iteration + 1\n, states_, buffer_, done\n) where:', _coconut_match_args)  #274 (line num in coconut source)

    buffer_, states_ = evaluate_policy(iteration, num_steps, agent, envs, states, buffer)  #278 (line num in coconut source)
    if len(buffer) > batch_size:  #280 (line num in coconut source)
        losses = update_policy(iteration, num_epochs, agent, buffer)  #281 (line num in coconut source)
    done = iteration >= num_iterations  #282 (line num in coconut source)
    _ = (_coconut_partial(write_performance, {1: iteration}, 2))(_coconut_iter_getitem(envs, 0))  #283 (line num in coconut source)

    return _coconut_tail_call(run_algorithm, envs, agent, episode, iteration + 1, states_, buffer_, done)  #285 (line num in coconut source)

def run_episode(agent: Agent, envs: gace.envs.vec.VecACE, episode: int):  #285 (line num in coconut source)
    obs = envs.reset()  #287 (line num in coconut source)
    keys = envs.info[0]  #288 (line num in coconut source)
    states = process_gace(obs, keys)  #289 (line num in coconut source)
    buffer = empty_buffer()  #290 (line num in coconut source)
    memory = run_algorithm(envs, agent, episode, 0, states, buffer, False)  #291 (line num in coconut source)
    _ = agent.save_state(model_path)  #292 (line num in coconut source)

    return agent  #293 (line num in coconut source)

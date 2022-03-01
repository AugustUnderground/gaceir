#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xf69e594e

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
from collections import namedtuple  #4 (line num in coconut source)
from itertools import repeat  #5 (line num in coconut source)
import torch as pt  #6 (line num in coconut source)
from torch.utils.data import TensorDataset  #7 (line num in coconut source)
from torch.utils.data import DataLoader  #7 (line num in coconut source)
import torch_optimizer as optim  #8 (line num in coconut source)
from torch.utils.tensorboard import SummaryWriter  #9 (line num in coconut source)
from fastprogress.fastprogress import master_bar  #10 (line num in coconut source)
from fastprogress.fastprogress import progress_bar  #10 (line num in coconut source)
import gym  #11 (line num in coconut source)
import gace  #11 (line num in coconut source)
import hace as ac  #12 (line num in coconut source)

## Defaults (as args later)
verbose: bool = True  # Print verbose debug output  #15 (line num in coconut source)
num_episodes: int = 666  # Number of episodes to play  #16 (line num in coconut source)
num_steps: int = 150  # How many steps to take per episode  #17 (line num in coconut source)
early_stop: float = -500.0  # Early stop criterion  #18 (line num in coconut source)
batch_size: int = 200  # size of the batches during epoch  #19 (line num in coconut source)
rng_seed: int = 666  # Random seed for reproducability  #20 (line num in coconut source)
algorithm: str = "std"  # Name of used algorithm ∈  ./algorithm  #21 (line num in coconut source)
ace_id: str = "op2"  # ACE Identifier of the Environment  #22 (line num in coconut source)
ace_backend: str = "xh035"  # PDK/Technology backend of the ACE Environment  #23 (line num in coconut source)
ace_variant: int = 0  # ACE Environment variant  #24 (line num in coconut source)
γ: float = 0.99  # Discount Factor  #25 (line num in coconut source)
τ_soft: float = 5e-3  # Avantage Factor  #26 (line num in coconut source)
α_actor: float = 2e-4  # Learning Rate for Actor  #27 (line num in coconut source)
α_critic: float = 1e-4  # Learning Rate for Critic  #28 (line num in coconut source)
α_value: float = 2e-4  # Learning Rate for Value  #29 (line num in coconut source)
β1: float = 0.9  # 0.9  #31 (line num in coconut source)
β2: float = 0.999  # 0.999  #32 (line num in coconut source)
βs: tuple[float] = (β1, β2)  # Lower Clipping  #33 (line num in coconut source)
σ_min: float = -20  # Lower Clipping  #33 (line num in coconut source)
σ_max: float = 2  # Upper Clipping  #34 (line num in coconut source)
buffer_size: int = (int)(1e7)  # Maximum size of replay buffer  #35 (line num in coconut source)
weights_init: float = 3e-3  # Weight Initializer  #36 (line num in coconut source)
max_time: float = 20.0  # Maximum time to cut off  #37 (line num in coconut source)
μ_gmoverid: float = 10.0  # Analog Design finally Automated  #38 (line num in coconut source)
μ_fug = 7.5  # Analog Design finally Automated  #39 (line num in coconut source)

## Setup
env_id: str = f"gace:{ace_id}-{ace_backend}-v{ace_variant}"  #42 (line num in coconut source)
time_stamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  #43 (line num in coconut source)
model_dir: str = f"./models/{time_stamp}-{env_id}-{algorithm}"  #44 (line num in coconut source)
model_path: str = f"{model_dir}/checkpoint.pt"  #45 (line num in coconut source)
log_dir: str = f"./runs/{time_stamp}-{env_id}-{algorithm}/"  #46 (line num in coconut source)
os.makedirs(model_dir, exist_ok=True)  #47 (line num in coconut source)

## Setup Globals
device = pt.device("cuda:1") if pt.cuda.is_available() else pt.device("cpu")  #50 (line num in coconut source)
writer = SummaryWriter(log_dir=log_dir, flush_secs=30)  #51 (line num in coconut source)
_ = (pt.manual_seed)(rng_seed)  #52 (line num in coconut source)

## Utility
def write_performance(env: gym.Env, step: int):  #55 (line num in coconut source)
    target = env.target  #56 (line num in coconut source)
    performance = (ac.current_performance)((env).ace)  #57 (line num in coconut source)
    for k in target.keys():  #58 (line num in coconut source)
        writer.add_scalars(k, {"performance": performance[k], "target": target[k]}, step)  #59 (line num in coconut source)

    return performance  #63 (line num in coconut source)

def save_checkpoint(model: Model, checkpoint_file: str):  #63 (line num in coconut source)
    state_dicts = (fmap)(_coconut.operator.methodcaller("state_dict"), [model.actor, model.critic_1, model.critic_2, model.value_online, model.value_target, model.actor_opt, model.critic_1_opt, model.critic_2_opt, model.value_opt])  #64 (line num in coconut source)
    keys = ["actor", "critic_1", "critic_2", "value_online", "value_target", "actor_opt", "critic_1_opt", "critic_2_opt", "value_opt"]  #68 (line num in coconut source)
    save_dict = (dict)((zip)(*(keys, state_dicts)))  #71 (line num in coconut source)
    res = (pt.save)(*(save_dict, checkpoint_file))  #72 (line num in coconut source)

## Prioritized Experience Replay Buffer
    return res  #75 (line num in coconut source)

class PERBuffer(_coconut.typing.NamedTuple("PERBuffer", [("priority", pt.Tensor), ("state", pt.Tensor), ("action", pt.Tensor), ("reward", pt.Tensor), ("next_state", pt.Tensor), ("done", pt.Tensor)])):  #75 (line num in coconut source)
    _coconut_is_data = True  #75 (line num in coconut source)
    __slots__ = ()  #75 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #75 (line num in coconut source)
    def __eq__(self, other):  #75 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #75 (line num in coconut source)
    def __hash__(self):  #75 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #75 (line num in coconut source)
    __match_args__ = ('priority', 'state', 'action', 'reward', 'next_state', 'done')  #75 (line num in coconut source)
    @_coconut_tco  #75 (line num in coconut source)
    def __len__(self):  #75 (line num in coconut source)
        return _coconut_tail_call((int), ((self.state).shape)[0])  #77 (line num in coconut source)

    def β_frame(self, idx: int, β_start: float=0.4, β_frames: int=(int)(1e5)):  #78 (line num in coconut source)
        fβ = min(1.0, (β_start + idx * (1.0 - β_start) / β_frames))  #80 (line num in coconut source)

        return fβ  #82 (line num in coconut source)

def make_per_buffer(state: pt.Tensor=pt.empty(0), action: pt.Tensor=pt.empty(0), reward: pt.Tensor=pt.empty(0), next_state: pt.Tensor=pt.empty(0), done: pt.Tensor=pt.empty(0)):  #82 (line num in coconut source)
    priority = (pt.ones_like)(reward)  #85 (line num in coconut source)
    buffer = (PERBuffer)(*(map)(_coconut.operator.methodcaller("to", device), (priority, state, action, reward, next_state, done)))  #86 (line num in coconut source)

    return buffer  #89 (line num in coconut source)

def push_per_buffer(buffer: PERBuffer, other: PERBuffer):  #89 (line num in coconut source)
    prio = (((pt.ones_like)(other.priority)).to(device)) * ((pt.max)(buffer.priority) if ((len)(buffer)) > 0 else 1.0)  #90 (line num in coconut source)
    buf_ = (PERBuffer)(*(map)(pt.cat, (_coconut_partial(zip, {0: buffer}, 2))((PERBuffer)(prio, *(map)(_coconut.operator.methodcaller("to", device), other[1:])))))  #94 (line num in coconut source)
    idx = (((_coconut_partial(pt.argsort, {}, 1, descending=True))((pt.squeeze)(buf_.priority)))[_coconut.slice(None, buffer_size)]).tolist()  #96 (line num in coconut source)
    buf = (PERBuffer)(*(map)(_coconut.operator.itemgetter((idx)), buf_))  #98 (line num in coconut source)

    return buf  #100 (line num in coconut source)

def update_per_buffer(buffer: PERBuffer, idx: list[int], priority: pt.Tensor):  #100 (line num in coconut source)
    _, state, action, reward, next_state, done = map(_coconut.operator.itemgetter((idx)), buffer)  #102 (line num in coconut source)
    prio = ((priority).to(device)).reshape(-1, 1)  #104 (line num in coconut source)
    buf_ = PERBuffer(prio, state, action, reward, next_state, done)  #105 (line num in coconut source)
    xdi = (list)(filter(lambda i: i not in idx, (range)((len)(buffer))))  #106 (line num in coconut source)
    buf = (PERBuffer)(*(map)(_coconut.operator.itemgetter((xdi)), buffer))  #107 (line num in coconut source)
    fsi = ((_coconut_partial(pt.argsort, {}, 1, descending=True))((pt.cat)((tuple)((map)(_coconut_forward_compose(_coconut.operator.attrgetter("priority"), pt.squeeze), (buf, buf_)))))).tolist()  #108 (line num in coconut source)
    buffer_ = (PERBuffer)(*(map)(_coconut_forward_compose(pt.cat, _coconut.operator.itemgetter((fsi))), zip(buf, buf_)))  #110 (line num in coconut source)

    return buffer_  #112 (line num in coconut source)

def sample_per_buffer(buffer: PERBuffer, batch_size: int, frame: int, α: float=0.6):  #112 (line num in coconut source)
    n = (len)(buffer)  #115 (line num in coconut source)
    probs = (_coconut_partial(pt.pow, {1: α}, 2))(buffer.priority)  #116 (line num in coconut source)
    p = (pt.squeeze)(probs / pt.sum(probs))  #117 (line num in coconut source)
    indices = ((_coconut.functools.partial(_coconut_iter_getitem, pt.arange(0, n)))(p.multinomial(num_samples=batch_size, replacement=False))).tolist()  #118 (line num in coconut source)
    smpl = (PERBuffer)(*(map)(_coconut.operator.itemgetter((indices)), buffer))  #120 (line num in coconut source)
    β = buffer.β_frame(frame)  #121 (line num in coconut source)
    w = pt.pow(n * p[indices], -β)  #122 (line num in coconut source)
    weights = w / pt.max(w)  #123 (line num in coconut source)

## Neural Networks
    return (smpl, indices, weights)  #126 (line num in coconut source)

class HStack(_coconut.collections.namedtuple("HStack", ()), pt.nn.Module):  #126 (line num in coconut source)
    _coconut_is_data = True  #126 (line num in coconut source)
    __slots__ = ()  #126 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #126 (line num in coconut source)
    def __eq__(self, other):  #126 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #126 (line num in coconut source)
    def __hash__(self):  #126 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #126 (line num in coconut source)
    __match_args__ = ()  #126 (line num in coconut source)
    @_coconut_tco  #126 (line num in coconut source)
    def forward(self, X: tuple[pt.Tensor, pt.Tensor]):  #126 (line num in coconut source)
        return _coconut_tail_call((pt.hstack), X)  #127 (line num in coconut source)


class VStack(_coconut.collections.namedtuple("VStack", ()), pt.nn.Module):  #129 (line num in coconut source)
    _coconut_is_data = True  #129 (line num in coconut source)
    __slots__ = ()  #129 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #129 (line num in coconut source)
    def __eq__(self, other):  #129 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #129 (line num in coconut source)
    def __hash__(self):  #129 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #129 (line num in coconut source)
    __match_args__ = ()  #129 (line num in coconut source)
    @_coconut_tco  #129 (line num in coconut source)
    def forward(self, X: tuple[pt.Tensor, pt.Tensor]):  #129 (line num in coconut source)
        return _coconut_tail_call((pt.vstack), X)  #130 (line num in coconut source)

## Actor Network

class ActorNet(_coconut.typing.NamedTuple("ActorNet", [("obs_dim", int), ("act_dim", int), ("σ_min", float), ("σ_max", float)]), pt.nn.Module):  #133 (line num in coconut source)
    _coconut_is_data = True  #133 (line num in coconut source)
    __slots__ = ()  #133 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #133 (line num in coconut source)
    def __eq__(self, other):  #133 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #133 (line num in coconut source)
    def __hash__(self):  #133 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #133 (line num in coconut source)
    __match_args__ = ('obs_dim', 'act_dim', 'σ_min', 'σ_max')  #133 (line num in coconut source)
    def __new__(_coconut_cls, obs_dim, act_dim, σ_min=-2.0, σ_max=20.0):  #133 (line num in coconut source)
        return _coconut.tuple.__new__(_coconut_cls, (obs_dim, act_dim, σ_min, σ_max))  #133 (line num in coconut source)
    def __init__(self, obs_dim: int, act_dim: int):  #135 (line num in coconut source)
        super(ActorNet, self).__init__()  #136 (line num in coconut source)
        self.lin_1 = pt.nn.Linear(obs_dim, 256)  #137 (line num in coconut source)
        self.lin_2 = pt.nn.Linear(256, 128)  #138 (line num in coconut source)
        self.lin_3 = pt.nn.Linear(128, 64)  #139 (line num in coconut source)
        self.lin_μ = pt.nn.Linear(64, act_dim)  #140 (line num in coconut source)
        self.lin_σ = pt.nn.Linear(64, act_dim)  #141 (line num in coconut source)
        self.relu = pt.nn.functional.relu  #142 (line num in coconut source)
        self.min_σ = σ_min  #143 (line num in coconut source)
        self.max_σ = σ_max  #144 (line num in coconut source)

    def forward(self, state: pt.Tensor):  #145 (line num in coconut source)
        x = (self.lin_3)((self.relu)((self.lin_2)((self.relu)((self.lin_1)(state)))))  #146 (line num in coconut source)
        μ = (self.lin_μ)(x)  #147 (line num in coconut source)
        σ = (_coconut_partial(pt.clamp, {1: self.min_σ, 2: self.max_σ}, 3))((self.lin_σ)(x))  #148 (line num in coconut source)

## Critic Network
        return (μ, σ)  #151 (line num in coconut source)

class CriticNet(_coconut.typing.NamedTuple("CriticNet", [("obs_dim", int), ("act_dim", int)]), pt.nn.Module):  #151 (line num in coconut source)
    _coconut_is_data = True  #151 (line num in coconut source)
    __slots__ = ()  #151 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #151 (line num in coconut source)
    def __eq__(self, other):  #151 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #151 (line num in coconut source)
    def __hash__(self):  #151 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #151 (line num in coconut source)
    __match_args__ = ('obs_dim', 'act_dim')  #151 (line num in coconut source)
    def __init__(self, obs_dim: int, act_dim: int):  #152 (line num in coconut source)
        super(CriticNet, self).__init__()  #153 (line num in coconut source)
        dim = obs_dim + act_dim  #154 (line num in coconut source)
        self.c_net = pt.nn.Sequential(HStack(), pt.nn.Linear(dim, 256), pt.nn.ReLU(), pt.nn.Linear(256, 128), pt.nn.ReLU(), pt.nn.Linear(128, 64), pt.nn.ReLU(), pt.nn.Linear(64, 1))  #155 (line num in coconut source)

    @_coconut_tco  #156 (line num in coconut source)
    def forward(self, state: pt.Tensor, action: pt.Tensor):  #156 (line num in coconut source)
        return _coconut_tail_call((self.c_net), (state, action))  #160 (line num in coconut source)

## Value Network

class ValueNet(_coconut.typing.NamedTuple("ValueNet", [("obs_dim", int)]), pt.nn.Module):  #163 (line num in coconut source)
    _coconut_is_data = True  #163 (line num in coconut source)
    __slots__ = ()  #163 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #163 (line num in coconut source)
    def __eq__(self, other):  #163 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #163 (line num in coconut source)
    def __hash__(self):  #163 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #163 (line num in coconut source)
    __match_args__ = ('obs_dim',)  #163 (line num in coconut source)
    def __init__(self, obs_dim: int):  #164 (line num in coconut source)
        super(ValueNet, self).__init__()  #165 (line num in coconut source)
        self.v_net = pt.nn.Sequential(pt.nn.Linear(obs_dim, 256), pt.nn.ReLU(), pt.nn.Linear(256, 128), pt.nn.ReLU(), pt.nn.Linear(128, 64), pt.nn.ReLU(), pt.nn.Linear(64, 1))  #166 (line num in coconut source)

    @_coconut_tco  #167 (line num in coconut source)
    def forward(self, state: pt.Tensor):  #167 (line num in coconut source)
        return _coconut_tail_call((self.v_net), state)  #170 (line num in coconut source)

## SAC Model

class Model(_coconut.typing.NamedTuple("Model", [("actor", pt.nn.Module), ("critic_1", pt.nn.Module), ("critic_2", pt.nn.Module), ("value_online", pt.nn.Module), ("value_target", pt.nn.Module), ("actor_opt", pt.optim.Optimizer), ("critic_1_opt", pt.optim.Optimizer), ("critic_2_opt", pt.optim.Optimizer), ("value_opt", pt.optim.Optimizer), ("critic_loss", _coconut.typing.Any), ("value_loss", _coconut.typing.Any)])):  #173 (line num in coconut source)
    _coconut_is_data = True  #173 (line num in coconut source)
    __slots__ = ()  #173 (line num in coconut source)
    __ne__ = _coconut.object.__ne__  #173 (line num in coconut source)
    def __eq__(self, other):  #173 (line num in coconut source)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  #173 (line num in coconut source)
    def __hash__(self):  #173 (line num in coconut source)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  #173 (line num in coconut source)
    __match_args__ = ('actor', 'critic_1', 'critic_2', 'value_online', 'value_target', 'actor_opt', 'critic_1_opt', 'critic_2_opt', 'value_opt', 'critic_loss', 'value_loss')  #173 (line num in coconut source)
    def soft_sync(self, other=None):  #173 (line num in coconut source)
        source = other if other else self.value_online  #179 (line num in coconut source)
        target = self.value_target  #180 (line num in coconut source)
        update = lambda t, o: (t.copy_)((t * (1.0 - τ_soft) + o * τ_soft))  #181 (line num in coconut source)
        _ = (list)((_coconut_partial(starmap, {0: update}, 2))((zip)(*(fmap)(_coconut_forward_compose(_coconut.operator.methodcaller("parameters"), list, _coconut.functools.partial(fmap, _coconut.operator.attrgetter("data"))), [target, source]))))  #182 (line num in coconut source)
        return self.value_target  #183 (line num in coconut source)

    def act(self, state: pt.Tensor):  #183 (line num in coconut source)
        μ, log_σ = (self.actor)(state)  #185 (line num in coconut source)
        σ = (pt.exp)(log_σ)  #186 (line num in coconut source)
        a = ((pt.tanh)(((pt.distributions.Normal)(*(μ, σ))).sample())).detach()  #187 (line num in coconut source)
        return a  #188 (line num in coconut source)

    def evaluate(self, state: pt.Tensor, ε: float=1e-6):  #188 (line num in coconut source)
        μ, log_σ = (self.actor)(state)  #190 (line num in coconut source)
        σ = (pt.exp)(log_σ)  #191 (line num in coconut source)
        n = pt.distributions.Normal(μ, σ)  #192 (line num in coconut source)
        z = n.sample()  #193 (line num in coconut source)
        a = (pt.tanh)(z)  #194 (line num in coconut source)
        ε_ = pt.log(1 - a.pow(2) + ε)  #195 (line num in coconut source)
        p = ((_coconut_partial((_coconut_minus), {1: ε_}, 2))((n).log_prob(z))).sum(-1, keepdim=True)  #196 (line num in coconut source)

        return (a, p, z, μ, log_σ)  #198 (line num in coconut source)

def make_model(act_dim: int, obs_dim: int):  #198 (line num in coconut source)
    def init_weights(m: pt.nn.Module, init: float):  #199 (line num in coconut source)
        if isinstance(m, pt.nn.Linear) and (m.out_features == 1 or m.out_features == act_dim):  #200 (line num in coconut source)
            m.weight.data.uniform_(-init, init)  #201 (line num in coconut source)
            m.bias.data.uniform_(-init, init)  #202 (line num in coconut source)
#critic_loss   = pt.nn.MSELoss()

    critic_loss = lambda qe, qt, w: (_coconut_partial((_coconut.operator.mul), {0: 0.5}, 2))((pt.mean)((_coconut_partial((_coconut.operator.mul), {1: w}, 2))((_coconut_partial(pt.pow, {1: 2}, 2))((qe - qt)))))  #204 (line num in coconut source)
    value_loss = pt.nn.MSELoss()  #206 (line num in coconut source)
    actor = (ActorNet(obs_dim, act_dim)).to(device)  #207 (line num in coconut source)
    critic_1 = (CriticNet(obs_dim, act_dim)).to(device)  #208 (line num in coconut source)
    critic_2 = (CriticNet(obs_dim, act_dim)).to(device)  #209 (line num in coconut source)
    value_online = (ValueNet(obs_dim)).to(device)  #210 (line num in coconut source)
    _ = [(n.apply)(_coconut_partial(init_weights, {1: weights_init}, 2)) for n in [actor, critic_1, critic_2, value_online]]  #211 (line num in coconut source)
    value_target = (ValueNet(obs_dim)).to(device)  #213 (line num in coconut source)
    params = (zip)(*(fmap)(_coconut.operator.methodcaller("parameters"), [value_target, value_online]))  #214 (line num in coconut source)
    _ = [(tp.data.copy_)((op).data) for tp, op in params]  #215 (line num in coconut source)
    actor_opt = pt.optim.Adam(actor.parameters(), lr=α_actor, betas=βs)  #216 (line num in coconut source)
    critic_1_opt = pt.optim.Adam(critic_1.parameters(), lr=α_critic, betas=βs)  #218 (line num in coconut source)
    critic_2_opt = pt.optim.Adam(critic_2.parameters(), lr=α_critic, betas=βs)  #220 (line num in coconut source)
    value_opt = pt.optim.Adam(value_online.parameters(), lr=α_value, betas=βs)  #222 (line num in coconut source)
    model = Model(actor, critic_1, critic_2, value_online, value_target, actor_opt, critic_1_opt, critic_2_opt, value_opt, critic_loss, value_loss)  #224 (line num in coconut source)

## Environment Step Post Processing
    return model  #229 (line num in coconut source)

def postprocess(observations, keys: dict[str, list[str]]):  #229 (line num in coconut source)
    pf = lambda k: ":" not in k and "/" not in k and (k[0]).islower()  #230 (line num in coconut source)
    lf = lambda k: k in ["ugbw", "cof", "sr_f", "sr_r"] or k.endswith(":fug")  #231 (line num in coconut source)
    p_idx = (list)((fmap)(keys["observations"].index, (filter)(pf, keys["observations"])))  #232 (line num in coconut source)
    a_idx = [(keys["observations"].index)(k) for k in keys["actions"]]  #235 (line num in coconut source)
    idx = (sorted)((p_idx + a_idx))  #236 (line num in coconut source)
    l_msk = ((pt.Tensor)((list)((_coconut_partial(map, {0: lf}, 2))(keys["observations"])))).to(device)  #237 (line num in coconut source)
    of = lambda o: (pt.where(o > 0, pt.log10(o), o) * l_msk) + (o * (1 - l_msk))  #238 (line num in coconut source)
    states = (pt.vstack)([((of)(((pt.from_numpy)(obs)).to(device)))[idx] for obs in observations])  #239 (line num in coconut source)
#states  = (pt.log10(states_) *  l_msk) + (states_ * (~l_msk))
#states = [ obs |> pt.from_numpy for obs in observations
#         ] |> pt.vstack |> .to(device)

    return states  #245 (line num in coconut source)

def scale_rewards(reward: pt.Tensor, ρ: float=1e-3):  #245 (line num in coconut source)
    scaled_reward = (reward - pt.mean(reward)) / (pt.std(reward) + ρ)  #246 (line num in coconut source)

## Update
    return scaled_reward  #249 (line num in coconut source)

def soft_q_update(epoch: int, model: Model, buffer: PERBuffer, batch_size: int, μλ: float=1e-3, σλ: float=1e-3, zλ: float=0.0, γ: float=0.99):  #249 (line num in coconut source)
    buffer_sample, idx, weights = (_coconut_partial(sample_per_buffer, {1: batch_size, 2: epoch}, 3))(buffer)  #252 (line num in coconut source)
    _, states, actions, rewards_, next_states, dones = (map)(_coconut.operator.methodcaller("to", device), buffer_sample)  #254 (line num in coconut source)
    rewards = (scale_rewards)(rewards_)  #256 (line num in coconut source)
#rewards     = rewards_
    new_actions, log_prob, z, μ, log_σ = (model.evaluate)(states)  #258 (line num in coconut source)
    q_exp_1 = (model.critic_1)(*(states, actions))  #260 (line num in coconut source)
    q_exp_2 = (model.critic_2)(*(states, actions))  #261 (line num in coconut source)
    v_expected = (model.value_online)(states)  #262 (line num in coconut source)
    v_target = (model.value_target)(next_states)  #263 (line num in coconut source)
    q_next = ((rewards + (1 - dones) * γ * v_target)).detach()  #264 (line num in coconut source)
#c1_loss     = (q_exp_1, q_next) |*> model.critic_loss
#c2_loss     = (q_exp_2, q_next) |*> model.critic_loss
    c1_loss = (model.critic_loss)(*(q_exp_1, q_next, weights))  #267 (line num in coconut source)
    c2_loss = (model.critic_loss)(*(q_exp_2, q_next, weights))  #268 (line num in coconut source)
    q_exp_1_nxt = (model.critic_1)(*(states, new_actions))  #269 (line num in coconut source)
    q_exp_2_nxt = (model.critic_2)(*(states, new_actions))  #270 (line num in coconut source)
    q_exp_nxt = (pt.min)(*(q_exp_1_nxt, q_exp_2_nxt))  #271 (line num in coconut source)
    v_next = ((q_exp_nxt - log_prob)).detach()  #272 (line num in coconut source)
    v_loss = (model.value_loss)(*(v_expected, v_next))  #273 (line num in coconut source)
    l_target = q_exp_nxt - v_expected  #274 (line num in coconut source)
    l_loss = (pt.mean)(((log_prob * (log_prob - l_target))).detach())  #275 (line num in coconut source)
    μ_loss = (_coconut_partial((_coconut.operator.mul), {0: μλ}, 2))((pt.mean)((_coconut_partial(pt.pow, {1: 2}, 2))(μ)))  #276 (line num in coconut source)
    σ_loss = (_coconut_partial((_coconut.operator.mul), {0: σλ}, 2))((pt.mean)((_coconut_partial(pt.pow, {1: 2}, 2))(log_σ)))  #277 (line num in coconut source)
    z_loss = (_coconut_partial((_coconut.operator.mul), {0: zλ}, 2))((pt.mean)(((_coconut_partial(pt.pow, {1: 2}, 2))(z)).sum(1)))  #278 (line num in coconut source)
    a_loss = l_loss + μ_loss + σ_loss + z_loss  #279 (line num in coconut source)
    _ = model.critic_1_opt.zero_grad()  #280 (line num in coconut source)
    _ = c1_loss.backward()  #281 (line num in coconut source)
    _ = model.critic_1_opt.step()  #282 (line num in coconut source)
    _ = model.critic_2_opt.zero_grad()  #283 (line num in coconut source)
    _ = c2_loss.backward()  #284 (line num in coconut source)
    _ = model.critic_2_opt.step()  #285 (line num in coconut source)
    _ = model.value_opt.zero_grad()  #286 (line num in coconut source)
    _ = v_loss.backward()  #287 (line num in coconut source)
    _ = model.value_opt.step()  #288 (line num in coconut source)
    _ = model.actor_opt.zero_grad()  #289 (line num in coconut source)
    _ = a_loss.backward()  #290 (line num in coconut source)
    _ = model.actor_opt.step()  #291 (line num in coconut source)
    _ = model.soft_sync()  #292 (line num in coconut source)
    td_1_err = ((q_next - q_exp_1)).detach()  #293 (line num in coconut source)
    td_2_err = ((q_next - q_exp_2)).detach()  #294 (line num in coconut source)
    priorities = (pt.squeeze)((pt.abs)((((td_1_err + td_2_err) / 2.0) + 1e-5)))  #295 (line num in coconut source)
    buffer_ = (_coconut_partial(update_per_buffer, {1: idx, 2: priorities}, 3))(buffer)  #296 (line num in coconut source)
    _ = writer.add_scalar("_Loss_Actor", a_loss, epoch)  #297 (line num in coconut source)
    _ = writer.add_scalar("_Loss_Critic_1", c1_loss, epoch)  #298 (line num in coconut source)
    _ = writer.add_scalar("_Loss_Critic_2", c2_loss, epoch)  #299 (line num in coconut source)
    _ = writer.add_scalar("_Loss_Value", v_loss, epoch)  #300 (line num in coconut source)

## Training Loop
    return buffer_  #303 (line num in coconut source)

@_coconut_mark_as_match  #303 (line num in coconut source)
def run_episode(*_coconut_match_args, **_coconut_match_kwargs):  #303 (line num in coconut source)
    _coconut_match_check_0 = False  #303 (line num in coconut source)
    _coconut_match_set_name_model = _coconut_sentinel  #303 (line num in coconut source)
    _coconut_match_set_name_episode = _coconut_sentinel  #303 (line num in coconut source)
    _coconut_match_set_name_buffer = _coconut_sentinel  #303 (line num in coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #303 (line num in coconut source)
    if (_coconut.len(_coconut_match_args) == 7) and ("model" not in _coconut_match_kwargs) and ("episode" not in _coconut_match_kwargs) and ("buffer" not in _coconut_match_kwargs) and (_coconut_match_args[4] is True):  #303 (line num in coconut source)
        _coconut_match_temp_0 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("model")  #303 (line num in coconut source)
        _coconut_match_temp_1 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("episode")  #303 (line num in coconut source)
        _coconut_match_temp_2 = _coconut_match_args[5] if _coconut.len(_coconut_match_args) > 5 else _coconut_match_kwargs.pop("buffer")  #303 (line num in coconut source)
        if not _coconut_match_kwargs:  #303 (line num in coconut source)
            _coconut_match_set_name_model = _coconut_match_temp_0  #303 (line num in coconut source)
            _coconut_match_set_name_episode = _coconut_match_temp_1  #303 (line num in coconut source)
            _coconut_match_set_name_buffer = _coconut_match_temp_2  #303 (line num in coconut source)
            _coconut_match_check_0 = True  #303 (line num in coconut source)
    if _coconut_match_check_0:  #303 (line num in coconut source)
        if _coconut_match_set_name_model is not _coconut_sentinel:  #303 (line num in coconut source)
            model = _coconut_match_temp_0  #303 (line num in coconut source)
        if _coconut_match_set_name_episode is not _coconut_sentinel:  #303 (line num in coconut source)
            episode = _coconut_match_temp_1  #303 (line num in coconut source)
        if _coconut_match_set_name_buffer is not _coconut_sentinel:  #303 (line num in coconut source)
            buffer = _coconut_match_temp_2  #303 (line num in coconut source)
    if not _coconut_match_check_0:  #303 (line num in coconut source)
        raise _coconut_FunctionMatchError('def run_episode(_, model, episode, _, True, buffer,_) = model where:', _coconut_match_args)  #303 (line num in coconut source)

    total = ((buffer.reward).sum()).item()  #304 (line num in coconut source)
    _ = writer.add_scalar(f"_Reward_Total", total, episode)  #305 (line num in coconut source)
    if verbose:  #306 (line num in coconut source)
        (print)(f"Episode {episode:03} Finished | Total Reward: {total}")  #307 (line num in coconut source)
    return model  #308 (line num in coconut source)

@_coconut_addpattern(run_episode)  #308 (line num in coconut source)
@_coconut_tco  #308 (line num in coconut source)
@_coconut_mark_as_match  #308 (line num in coconut source)
def run_episode(*_coconut_match_args, **_coconut_match_kwargs):  #308 (line num in coconut source)
    _coconut_match_check_1 = False  #308 (line num in coconut source)
    _coconut_match_set_name_envs = _coconut_sentinel  #308 (line num in coconut source)
    _coconut_match_set_name_model = _coconut_sentinel  #308 (line num in coconut source)
    _coconut_match_set_name_episode = _coconut_sentinel  #308 (line num in coconut source)
    _coconut_match_set_name_iteration = _coconut_sentinel  #308 (line num in coconut source)
    _coconut_match_set_name_finish = _coconut_sentinel  #308 (line num in coconut source)
    _coconut_match_set_name_buffer = _coconut_sentinel  #308 (line num in coconut source)
    _coconut_match_set_name_states = _coconut_sentinel  #308 (line num in coconut source)
    _coconut_FunctionMatchError = _coconut_get_function_match_error()  #308 (line num in coconut source)
    if (_coconut.len(_coconut_match_args) <= 7) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "envs" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "model" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "episode" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 3, "iteration" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 4, "finish" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 5, "buffer" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 6, "states" in _coconut_match_kwargs)) == 1):  #308 (line num in coconut source)
        _coconut_match_temp_3 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("envs")  #308 (line num in coconut source)
        _coconut_match_temp_4 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("model")  #308 (line num in coconut source)
        _coconut_match_temp_5 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("episode")  #308 (line num in coconut source)
        _coconut_match_temp_6 = _coconut_match_args[3] if _coconut.len(_coconut_match_args) > 3 else _coconut_match_kwargs.pop("iteration")  #308 (line num in coconut source)
        _coconut_match_temp_7 = _coconut_match_args[4] if _coconut.len(_coconut_match_args) > 4 else _coconut_match_kwargs.pop("finish")  #308 (line num in coconut source)
        _coconut_match_temp_8 = _coconut_match_args[5] if _coconut.len(_coconut_match_args) > 5 else _coconut_match_kwargs.pop("buffer")  #308 (line num in coconut source)
        _coconut_match_temp_9 = _coconut_match_args[6] if _coconut.len(_coconut_match_args) > 6 else _coconut_match_kwargs.pop("states")  #308 (line num in coconut source)
        if not _coconut_match_kwargs:  #308 (line num in coconut source)
            _coconut_match_set_name_envs = _coconut_match_temp_3  #308 (line num in coconut source)
            _coconut_match_set_name_model = _coconut_match_temp_4  #308 (line num in coconut source)
            _coconut_match_set_name_episode = _coconut_match_temp_5  #308 (line num in coconut source)
            _coconut_match_set_name_iteration = _coconut_match_temp_6  #308 (line num in coconut source)
            _coconut_match_set_name_finish = _coconut_match_temp_7  #308 (line num in coconut source)
            _coconut_match_set_name_buffer = _coconut_match_temp_8  #308 (line num in coconut source)
            _coconut_match_set_name_states = _coconut_match_temp_9  #308 (line num in coconut source)
            _coconut_match_check_1 = True  #308 (line num in coconut source)
    if _coconut_match_check_1:  #308 (line num in coconut source)
        if _coconut_match_set_name_envs is not _coconut_sentinel:  #308 (line num in coconut source)
            envs = _coconut_match_temp_3  #308 (line num in coconut source)
        if _coconut_match_set_name_model is not _coconut_sentinel:  #308 (line num in coconut source)
            model = _coconut_match_temp_4  #308 (line num in coconut source)
        if _coconut_match_set_name_episode is not _coconut_sentinel:  #308 (line num in coconut source)
            episode = _coconut_match_temp_5  #308 (line num in coconut source)
        if _coconut_match_set_name_iteration is not _coconut_sentinel:  #308 (line num in coconut source)
            iteration = _coconut_match_temp_6  #308 (line num in coconut source)
        if _coconut_match_set_name_finish is not _coconut_sentinel:  #308 (line num in coconut source)
            finish = _coconut_match_temp_7  #308 (line num in coconut source)
        if _coconut_match_set_name_buffer is not _coconut_sentinel:  #308 (line num in coconut source)
            buffer = _coconut_match_temp_8  #308 (line num in coconut source)
        if _coconut_match_set_name_states is not _coconut_sentinel:  #308 (line num in coconut source)
            states = _coconut_match_temp_9  #308 (line num in coconut source)
    if not _coconut_match_check_1:  #308 (line num in coconut source)
        raise _coconut_FunctionMatchError('addpattern def run_episode( envs, model, episode, iteration, finish, buffer, states\n) = run_episode( envs, model, episode, iteration_\n, finish_, buffer_, states_ ) where:', _coconut_match_args)  #308 (line num in coconut source)

    t0 = time.time()  #311 (line num in coconut source)
    actions = (model.act)(states)  #312 (line num in coconut source)
    observations, rewards_, dones_, infos = (envs.step)((list)((fmap)(_coconut_forward_compose(_coconut.operator.methodcaller("squeeze"), _coconut.operator.methodcaller("cpu"), _coconut.operator.methodcaller("numpy")), (_coconut_partial(pt.split, {1: 1}, 2))(actions))))  #313 (line num in coconut source)
    states_ = postprocess(observations, envs.info[0])  #317 (line num in coconut source)
    rewards = (((pt.Tensor)(rewards_)).to(device)).reshape(-1, 1)  #318 (line num in coconut source)
    dones = (((pt.Tensor)(dones_)).to(device)).reshape(-1, 1)  #319 (line num in coconut source)
    buf = (_coconut_partial(push_per_buffer, {0: buffer}, 2))((make_per_buffer)(*(states, actions, rewards, states_, dones)))  #320 (line num in coconut source)
    t1 = time.time()  #322 (line num in coconut source)
    dt = t1 - t0  #323 (line num in coconut source)
    _ = (_coconut_partial(write_performance, {1: iteration}, 2))(_coconut_iter_getitem(envs, 0))  #324 (line num in coconut source)
    buffer_ = (soft_q_update(iteration, model, buf, batch_size) if len(buf) > batch_size else buf)  #325 (line num in coconut source)
    done_ = (_coconut_forward_compose(pt.squeeze, _coconut.operator.methodcaller("bool"), pt.all, _coconut.operator.methodcaller("item")))(dones)  #328 (line num in coconut source)
    stop_ = (((pt.mean)(rewards)).item()) < early_stop  #329 (line num in coconut source)
    finish_ = done_ or stop_ or (iteration >= num_steps)  # or (dt > max_time)  #330 (line num in coconut source)
    iteration_ = iteration + 1  #331 (line num in coconut source)
    if verbose:  #332 (line num in coconut source)
        (print)(f"Iteration {iteration:03} took {dt:.3f}s | Average Reward: {rewards.mean():.3f}")  #333 (line num in coconut source)
    _ = writer.add_scalar("_Reward_Mean", rewards.mean(), iteration)  #334 (line num in coconut source)
    _ = save_checkpoint(model, model_path)  #335 (line num in coconut source)

## Episode Loop
    return _coconut_tail_call(run_episode, envs, model, episode, iteration_, finish_, buffer_, states_)  #338 (line num in coconut source)

def run_episodes(model: Model, envs: gace.envs.vec.VecACE, episode: int):  #338 (line num in coconut source)
    obs = envs.reset()  #340 (line num in coconut source)
    states = postprocess(obs, envs.info[0])  #341 (line num in coconut source)
#buffer  = pt.empty(0) |> .to(device) |> repeat |> .$[:5] |> tuple |*> ReplayBuffer
    buffer = make_per_buffer()  #343 (line num in coconut source)
    model = run_episode(envs, model, episode, 0, False, buffer, states)  #344 (line num in coconut source)

    return model  #345 (line num in coconut source)

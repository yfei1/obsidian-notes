# Python Import Binding: `from import` vs Module Attribute Access
#ml-systems #interview-prep

**Scope**: Python name-binding mechanics as they apply to monkey-patching patterns in ML serving stacks — specifically, closures that must read a module-level variable set after import time.

**Prerequisites**: Familiarity with Python modules and closures. No ML-specific prerequisites.

## TL;DR

`from module import name` creates a **local binding** that snapshots the value at execution
time. If the module later mutates `name`, the local binding stays stale. `import module;
module.name` always reads the current value through attribute lookup on the module object.
This distinction is critical for monkey-patching patterns (runtime replacement of a function or attribute in an already-imported module) where a module-level variable
(like a process group handle — an object coordinating communication across distributed workers) starts as `None` and gets set later at runtime.

---

## Core Intuition

The problem: you need a closure that reads a module-level variable that doesn't exist yet.

```python
# afm_pt_moe.py (simplified)
_PT = None          # None at import time

def init_groups():  # called later during model __init__
    global _PT
    _PT = GroupCoordinator(...)  # GroupCoordinator: object managing cross-worker communication channels
```

A monkey-patch installed at plugin load time needs to read `_PT` at runtime (after
`init_groups()` runs). The wrong import style freezes `_PT` to `None` forever.

---

## How It Works

Python's `import` statement has two distinct behaviors depending on syntax.

### `from module import name` — copies the reference

```python
from afm_pt_moe import _PT
```

This is equivalent to:

```python
import afm_pt_moe
_PT = afm_pt_moe._PT   # local variable = current value (None)
del afm_pt_moe          # module ref discarded
```

The local `_PT` is a **new name binding** in the current namespace. It points to whatever
object `afm_pt_moe._PT` referenced at that moment (`None`). Concretely:

```python
import sys, types

# Simulate the module
mod = types.ModuleType("afm_pt_moe")
mod._PT = None
sys.modules["afm_pt_moe"] = mod

# from-import: snapshot the reference
local_PT = mod._PT          # local_PT -> id(None)
print(id(local_PT))         # e.g. 4298409432  (CPython singleton)
print(id(mod._PT))          # same id — both point to None

# Now rebind the module-level name
mod._PT = object()          # new object, new id
print(id(local_PT))         # 4298409432 — unchanged
print(id(mod._PT))          # e.g. 4372951120 — different object
# local_PT and mod._PT now point to different objects
```

When `init_groups()` later rebinds `afm_pt_moe._PT` to a `GroupCoordinator`, the local
`_PT` still points to `None` because rebinding a name in one namespace does not affect
bindings in another namespace.

### `import module` — keeps the module object

```python
import afm_pt_moe as pt_mod
pt_mod._PT  # attribute lookup on the module object
```

`pt_mod` is a reference to the **module object** itself. `pt_mod._PT` is an attribute
lookup that reads `pt_mod.__dict__["_PT"]` at access time — a single dict key lookup.
When `init_groups()` rebinds `afm_pt_moe._PT`, it writes into that same `__dict__`,
so all subsequent `pt_mod._PT` reads see the new value:

```python
mod._PT = None
print(mod.__dict__["_PT"])  # None

mod._PT = object()           # rebind in __dict__
print(mod.__dict__["_PT"])  # new object — same dict, updated value
```

### Where it matters: closures vs function bodies

| Pattern | Binding time | Sees mutations? |
|---------|-------------|-----------------|
| `from mod import x` at **module level** | Once, at import | No |
| `from mod import x` inside **function body** | Each call | Yes (re-executes import machinery) |
| `import mod; mod.x` inside **function body** | Each call (attribute lookup) | Yes (lightweight dict lookup) |

A `from import` inside a function body technically works because it re-executes each call.
But it goes through the full import machinery (`sys.modules` lookup, frame construction,
name binding) rather than a single `__dict__` key access. Module attribute access (`mod.x`)
is a plain dict lookup — and unambiguous about intent.

---

## Proof: Two Namespaces, Two Fates

Self-contained script (stdlib only, Python 3.8+). All assertions pass:

```python
import sys, types

def make_module():
    """Fresh module with _PT = None and an init_groups() that rebinds it."""
    mod = types.ModuleType("_test_mod")
    mod._PT = None
    def init_groups():
        mod._PT = "LIVE_VALUE"
    mod.init_groups = init_groups
    sys.modules["_test_mod"] = mod
    return mod

# Test 1: from-import at module level → snapshot (frozen at None)
mod = make_module()
local_PT = mod._PT          # equivalent to: from _test_mod import _PT
mod.init_groups()            # rebinds mod._PT → "LIVE_VALUE"
assert local_PT is None      # PASS — local binding still points to None
assert mod._PT == "LIVE_VALUE"  # PASS — module dict updated

# Test 2: module attribute access → live
mod = make_module()
mod.init_groups()
assert mod._PT == "LIVE_VALUE"  # PASS — reads current __dict__ value

# Test 3: from-import inside function → also works (re-executes each call)
mod = make_module()
mod.init_groups()
def f():
    return sys.modules["_test_mod"]._PT   # equivalent to from-import inside fn
assert f() == "LIVE_VALUE"   # PASS — re-reads on each call

# Test 4: closure over module-level binding → frozen
mod = make_module()
captured = mod._PT           # snapshot at closure-definition time
def g():
    return captured
mod.init_groups()
assert g() is None           # PASS — closure sees original None forever

# Test 5: id() confirms two distinct objects after rebind
mod = make_module()
id_before = id(mod._PT)      # id of None singleton
local_snap = mod._PT         # snapshot
mod.init_groups()            # rebind
assert id(local_snap) == id_before   # PASS — snapshot unchanged
assert id(mod._PT) != id_before      # PASS — module now points elsewhere

print("All 5 assertions pass")
```

```
All 5 assertions pass
```

<!-- verify
import sys, types

def make_module():
    mod = types.ModuleType("_test_mod")
    mod._PT = None
    def init_groups(): mod._PT = "LIVE_VALUE"
    mod.init_groups = init_groups
    sys.modules["_test_mod"] = mod
    return mod

mod = make_module(); local_PT = mod._PT; mod.init_groups()
assert local_PT is None
assert mod._PT == "LIVE_VALUE"
mod = make_module(); mod.init_groups()
assert mod._PT == "LIVE_VALUE"
mod = make_module(); mod.init_groups()
def f(): return sys.modules["_test_mod"]._PT
assert f() == "LIVE_VALUE"
mod = make_module(); captured = mod._PT
def g(): return captured
mod.init_groups(); assert g() is None
mod = make_module(); id_before = id(mod._PT); local_snap = mod._PT; mod.init_groups()
assert id(local_snap) == id_before
assert id(mod._PT) != id_before
print("verify: all assertions pass")
-->

---

## The Real Pattern: vLLM Plugin Monkey-Patching

Our PT-MoE plugin patches `graph_capture()` (the hook vLLM calls before CUDA graph recording — capturing a replayable sequence of GPU ops to avoid Python overhead on each forward pass) at plugin load time. `_PT` is `None` then:

```python
# _vllm_plugin.py — installed at plugin load (Phase 1)
@contextmanager
def patched_graph_capture(device):
    # GOOD: module attribute access — reads _PT at call time (Phase 7)
    import ray_vllm_extension.models.afm_pt_moe as pt_mod
    pt_group = pt_mod._PT  # reads current value

    # BAD: from-import at closure scope — would snapshot None forever
    # from ray_vllm_extension.models.afm_pt_moe import _PT
    # pt_group = _PT  # always None
```

Timeline: plugin loads (Phase 1: startup) → `_PT = None` → model init sets `_PT` (Phase 5: weight loading) →
graph capture reads `pt_mod._PT` (Phase 7: CUDA graph capture) → gets the live `GroupCoordinator`.

Source: `_vllm_plugin.py:169-175`

---

## When Each Pattern Is Correct

### When to use module attribute access (`mod.x`)

Use `mod.x` whenever the variable may be rebound after your code's import time — because `from import` snapshots the reference once, so any later rebind in the module's `__dict__` is invisible to your local binding.

- **Deferred initialization** (e.g., `_PT = None` → `GroupCoordinator` later): the variable's final value doesn't exist yet at import time, so a snapshot is always stale — because the snapshot is taken before `init_groups()` runs
- **Monkey-patches and plugin callbacks**: these execute before the target variable is set by definition, so the snapshot would always capture the sentinel value — because plugin load (Phase 1) precedes model init (Phase 5)
- **Hot-path function calls**: `mod.x` is a single `__dict__` key lookup (1 dict read: `module.__dict__["x"]`), while `from import` inside a function re-runs the full import machinery on every call: `sys.modules` hash lookup → `LOAD_ATTR` on the module → `STORE_FAST` into the local frame — because Python re-executes the import statement each time the function body runs

### When `from import` is fine

Use `from import` when the snapshot taken at import time will always match the live value — because no rebind will occur after that point.

- **Classes, functions, and constants**: these are defined once and never rebound — because module-level `def` and `class` statements execute once at import and the name is never reassigned
- **Import inside a function body**: re-executes on each call, so the snapshot is always fresh — because the `from import` statement runs again, reading the current `__dict__` value each time (though `mod.x` is cheaper for the same result)
- **Import order guarantees**: if the import happens after the mutation (e.g., the module is fully initialized before any other module imports from it) — because the snapshot captures the final value

### The mutable object workaround

If the variable points to a **mutable container** (list, dict), `from import` works because both names reference the same object — mutations to the object are visible through either name. The problem is specifically with **rebinding** (reassigning the name to a new object), not mutating the existing object.

```python
# Mutable container — from-import works
registry = {}
from mod import registry   # both point to same dict
registry["key"] = "val"    # visible through both names — same object, mutated in place

# Rebinding — from-import breaks
_PT = None
from mod import _PT        # local = None
_PT = GroupCoordinator()   # only rebinds module-level name, not local binding
```

---

## See Also

This note is the canonical reference for Python import binding in the vault. Notes that use monkey-patching or deferred initialization patterns should link here rather than re-explain the `from import` vs `mod.x` distinction.

- [[ml-systems/vllm/vllm-model-integration]] — where the monkey-patching pattern is used
- [[ml-systems/vllm/pt-moe-vllm-implementation]] — the PT-MoE integration that requires this pattern
- [[ml-systems/distributed/vllm-distributed-groups]] — process group lifecycle that `_PT` participates in
- [[ml-systems/gpu/pytorch-module-hooks]] — `nn.Module.__call__` aliasing to `_wrapped_call_impl` is a direct instance of the attribute binding mechanics described here

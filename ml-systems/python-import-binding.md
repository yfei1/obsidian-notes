# Python Import Binding: `from import` vs Module Attribute Access
#ml-systems #interview-prep

## TL;DR

`from module import name` creates a **local binding** that snapshots the value at execution
time. If the module later mutates `name`, the local binding stays stale. `import module;
module.name` always reads the current value through attribute lookup on the module object.
This distinction is critical for monkey-patching patterns where a module-level variable
(like a process group handle) starts as `None` and gets set later at runtime.

---

## Core Intuition

The problem: you need a closure that reads a module-level variable that doesn't exist yet.

```python
# afm_pt_moe.py (simplified)
_PT = None          # None at import time

def init_groups():  # called later during model __init__
    global _PT
    _PT = GroupCoordinator(...)
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

## Concrete Proof

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

---

## The Real Pattern: vLLM Plugin Monkey-Patching

Our PT-MoE plugin patches `graph_capture()` at plugin load time. `_PT` is `None` then:

```python
# _vllm_plugin.py — installed at plugin load (Phase 1)
@contextmanager
def patched_graph_capture(device):
    # GOOD: module attribute access — reads _PT at call time (Phase 7)
    import apple_ray_vllm_extension.models.afm_pt_moe as pt_mod
    pt_group = pt_mod._PT  # reads current value

    # BAD: from-import at closure scope — would snapshot None forever
    # from apple_ray_vllm_extension.models.afm_pt_moe import _PT
    # pt_group = _PT  # always None
```

Timeline: plugin loads (Phase 1) → `_PT = None` → model init sets `_PT` (Phase 5) →
graph capture reads `pt_mod._PT` (Phase 7) → gets the live `GroupCoordinator`.

Source: `_vllm_plugin.py:169-175`

---

## Key Trade-offs & Decisions

**When to use module attribute access (`mod.x`):**
- The variable is mutated after your code's import time
- Monkey-patches, plugin callbacks, lazy initialization patterns
- Performance matters (dict lookup vs import machinery)

**When `from import` is fine:**
- The imported name is a class, function, or constant that never changes
- You import inside a function body and accept the import overhead
- The binding happens after the mutation (import order guarantees it)

**The mutable object workaround:**
If the variable points to a mutable container (list, dict), `from import` works because
both names reference the same object and mutations are visible through either name. The
problem is specifically with **rebinding** (reassigning the name), not mutating the object.

```python
# Mutable container — from-import works
registry = {}
from mod import registry   # both point to same dict
registry["key"] = "val"    # visible through both names

# Rebinding — from-import breaks
_PT = None
from mod import _PT        # local = None
_PT = GroupCoordinator()   # only rebinds module-level, not local
```

---

## See Also

- [[ml-systems/vllm-model-integration]] — where the monkey-patching pattern is used
- [[ml-systems/pt-moe-vllm-implementation]] — the PT-MoE integration that requires this pattern
- [[ml-systems/vllm-distributed-groups]] — process group lifecycle that _PT participates in

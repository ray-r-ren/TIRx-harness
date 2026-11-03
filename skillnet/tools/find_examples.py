"""find_kernel_examples: keyword search over example kernels and the TIRx-lite reference."""

from __future__ import annotations

import ast
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from skillnet_sdk import OutputFile, ToolContext, ToolError, ToolInput, ToolOutput, ToolResult, tool

from ._common import REFDOCS, TOOLS_DIR
from ._standalone import make_standalone, package_of

STARTERS = TOOLS_DIR / "data" / "starters"
STARTER_NAMES = ("elementwise", "saxpy", "reduction", "softmax", "rmsnorm", "layernorm", "matmul")
# Words users say for each starter; a query containing one ranks that starter first.
STARTER_ALIASES = {
    "elementwise": ["elementwise", "element-wise", "element wise", "pointwise", "vector add", "vector_add", "vectoradd",
                    "vadd", "add", "addition", "relu", "scale", "map"],
    "saxpy": ["saxpy", "axpy", "a*x+y", "alpha*x+y", "alpha x plus y"],
    "reduction": ["reduction", "reduce", "sum", "row sum", "row_sum", "rowsum", "row-wise sum"],
    "softmax": ["softmax", "soft max"],
    "rmsnorm": ["rmsnorm", "rms norm", "rms_norm", "rms-norm", "root mean square norm"],
    "layernorm": ["layernorm", "layer norm", "layer_norm", "layer-norm", "layer normalization"],
    "matmul": ["matmul", "mat mul", "gemm", "sgemm", "matrix multiply", "matrix multiplication", "matrix product",
               "matrix-multiply", "mm"],
}
SKIP_KERNEL_DIRS = {"tirx_lite", "test", "bench", "bench_suite", "__pycache__"}
SKIP_KERNEL_FILES = {"__init__.py", "_protocol.py", "registry.py", "runner.py", "reference_requirements.py"}

# Words users say -> words the sources use (searched with a lower weight).
SYNONYMS = {
    "matmul": ["gemm"], "gemm": ["matmul"], "mm": ["gemm"],
    "norm": ["rmsnorm", "layernorm"], "normalization": ["rmsnorm", "norm"], "layernorm": ["norm"],
    "rmsnorm": ["norm", "rsqrt"],
    "attention": ["attn", "flash", "mla", "mqa"], "attn": ["attention"],
    "reduction": ["reduce", "sum", "shfl"], "reduce": ["sum", "shfl"], "sum": ["reduce"],
    "shuffle": ["shfl"], "shfl": ["shuffle"], "warp": ["lane"],
    "barrier": ["bar", "mbarrier", "sync"], "sync": ["barrier", "bar"], "mbarrier": ["barrier"],
    "shared": ["smem"], "smem": ["shared"], "global": ["gmem"],
    "add": ["vector_add", "elementwise"], "elementwise": ["vector_add", "saxpy"], "axpy": ["saxpy"],
    "softmax": ["exp", "max"], "load": ["ld"], "store": ["st"], "ld": ["load"], "st": ["store"],
    "half": ["f16"], "fp16": ["f16"], "bf16": ["bfloat16"], "float": ["f32"], "fp32": ["f32"],
    "tensor": ["wgmma", "mma"], "tma": ["cp", "async", "bulk"], "moe": ["grouped"],
    "loop": ["serial", "unroll"], "if": ["select"], "branch": ["if", "select"],
}
STOP = {"a", "an", "the", "for", "of", "and", "or", "to", "in", "on", "with", "kernel", "kernels", "i", "need",
        "how", "do", "write", "use", "my", "is", "it", "that", "this", "from", "by", "as", "be", "what"}


class FindIn(ToolInput):
    query: str = Field(
        default="", max_length=200,
        description="Math or operation words to look for, e.g. 'rmsnorm', 'row sum warp shuffle', 'mbarrier', "
                    "'ld.global f16', 'gemm', 'bar.sync'. Required unless `starter` is set.",
    )
    scope: Literal["kernels", "reference", "both"] = Field(
        default="both",
        description="What to search: 'kernels' (example kernel sources), 'reference' (TIRx-lite manuals, "
                    "checker/simulator guides and the simulator's supported-operations table) or 'both'.",
    )
    max_results: int = Field(default=8, ge=1, le=20, description="How many matches to return (1-20).")
    starter: Literal["elementwise", "saxpy", "reduction", "softmax", "rmsnorm", "layernorm", "matmul"] | None = Field(
        default=None,
        description="Return exactly this verified H100 starter kernel (ignores query): elementwise ('vector add'), "
                    "saxpy, reduction ('sum'), softmax, rmsnorm ('RMS norm'), layernorm ('layer norm'), matmul "
                    "('GEMM', 'matrix multiply').",
    )


class Match(BaseModel):
    kind: Literal["kernel", "api", "doc"] = Field(description="kernel = example kernel source; api = TIRx-lite "
                                                             "library source; doc = reference section")
    name: str
    path: str = Field(description="Where it lives: starter/<file>, tirx_kernels/<path> or reference/<doc>#<section>")
    archs: list[str] = Field(default_factory=list, description="GPU architectures the kernel targets (kernels only)")
    runs_on_h100: bool | None = Field(default=None, description="Whether the kernel can be timed on this skill's "
                                                                "H100 (sm_90a); Blackwell sm_100a kernels cannot")
    verified: bool = Field(default=False, description="Starter kernels: simulated and race-checked clean")
    standalone: bool | None = Field(default=None, description="The returned .py has no relative imports and every "
                                                              "module it imports from tirx_kernels exists")
    excerpt: str
    score: float
    source: str | None = Field(default=None, description="Full source text (top 2 starter matches): pass it as "
                                                         "`source` to inspect_kernel / simulate_kernel")
    file: str | None = Field(default=None, description="Name of the output .py holding this source")


class FindOut(ToolOutput):
    query: str
    matches: list[Match]
    total: int = Field(description="How many items matched before the max_results cut")
    attached_files: list[str] = Field(description="Names of the source and reference files returned as outputs")


@dataclass
class Item:
    kind: str
    name: str
    path: str
    file: Path
    text: str
    head: str  # name, path and docstring: weighted higher
    archs: list[str] = field(default_factory=list)
    verified: bool = False
    lines: list[str] = field(default_factory=list)
    package: str | None = None  # dotted package for collection files (to resolve relative imports)
    starter: str | None = None


_TOKEN = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")


def tokens(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        out.append(t)
        if "." in t or "_" in t:
            out.extend(p for p in re.split(r"[._]", t) if p)
    return out


def _docstring(src: str) -> str:
    try:
        return ast.get_docstring(ast.parse(src)) or ""
    except SyntaxError:
        return ""


def _kernel_meta(src: str) -> dict:
    m = re.search(r"^KERNEL_META\s*=\s*(\{.*?\})\s*$", src, re.M | re.S)
    if not m:
        return {}
    try:
        meta = ast.literal_eval(m.group(1))
        return meta if isinstance(meta, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections, title, buf = [], "", []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,3}\s", line):
            if buf and "".join(buf).strip():
                sections.append((title, "\n".join(buf)))
            title, buf = line.lstrip("#").strip(), [line]
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append((title, "\n".join(buf)))
    return sections


_CORPUS: list[Item] | None = None


def corpus() -> list[Item]:
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    items: list[Item] = []
    for stem in STARTER_NAMES:
        f = STARTERS / f"{stem}.py"
        src = f.read_text(encoding="utf-8")
        doc = _docstring(src)
        items.append(Item("kernel", f.stem, f"starter/{f.name}", f, src, f"{f.stem} {doc}", ["sm_90a"], True,
                          src.splitlines(), None, stem))
    try:
        import tirx_kernels

        kroot = Path(tirx_kernels.__file__).resolve().parent
    except ImportError:
        kroot = None
    if kroot is not None:
        for f in sorted(kroot.rglob("*.py")):
            rel = f.relative_to(kroot)
            if f.name in SKIP_KERNEL_FILES and len(rel.parts) == 1:
                continue
            if rel.parts[0] == "tirx_lite":
                if f.name == "__init__.py":
                    continue
                src = f.read_text(encoding="utf-8", errors="replace")
                items.append(Item("api", f"tirx_lite.{f.stem}", f"tirx_kernels/{rel}", f, src,
                                  f"tirx_lite {f.stem} {_docstring(src)[:600]}", [], False, src.splitlines(),
                                  package_of(rel)))
                continue
            if any(p in SKIP_KERNEL_DIRS for p in rel.parts[:-1]) or f.name == "__init__.py":
                continue
            src = f.read_text(encoding="utf-8", errors="replace")
            if "tirx_lite" not in src and "KERNEL_META" not in src:
                continue
            meta = _kernel_meta(src)
            archs = [str(a) for a in meta.get("runtime_cuda_archs", []) if isinstance(a, str)]
            name = str(meta.get("name") or f.stem)
            head = f"{name} {rel.with_suffix('')} {meta.get('category', '')} {_docstring(src)[:600]}"
            items.append(Item("kernel", name, f"tirx_kernels/{rel}", f, src, head, archs, False, src.splitlines(),
                              package_of(rel)))
        for f in sorted(kroot.rglob("*.md")):
            _add_doc(items, f, f"tirx_kernels/{f.relative_to(kroot)}")
    for f in sorted(REFDOCS.glob("*.md")):
        _add_doc(items, f, f"reference/{f.name}")
    try:
        import tirx_harness

        ops = Path(tirx_harness.__file__).resolve().parent / "numsim" / "engine-rs" / "SUPPORTED_OPS.md"
        if ops.exists():
            _add_doc(items, ops, "reference/numsim-SUPPORTED_OPS.md")
    except ImportError:
        pass
    _CORPUS = items
    return items


def _add_doc(items: list[Item], f: Path, label: str) -> None:
    text = f.read_text(encoding="utf-8", errors="replace")
    for title, body in _split_sections(text):
        name = f"{f.stem}: {title}" if title else f.stem
        items.append(Item("doc", name, f"{label}#{title}" if title else label, f, body, f"{f.stem} {title}",
                          [], False, body.splitlines()))


def _excerpt(item: Item, terms: set[str], limit: int = 420) -> str:
    lines = item.lines
    best, best_hits = 0, -1
    for i, line in enumerate(lines):
        low = line.lower()
        hits = sum(1 for t in terms if t in low)
        if hits > best_hits:
            best, best_hits = i, hits
    if item.kind == "kernel":
        doc = _docstring(item.text)
        if doc and (best_hits <= 0 or any(t in doc.lower() for t in terms)):
            return doc[:limit]
    lo = max(0, best - 2)
    snippet = "\n".join(lines[lo:lo + 7]).strip()
    return snippet[:limit]


def starter_aliases(query: str) -> list[str]:
    """Starters whose user words appear in the query, in order of first appearance."""
    q = " " + re.sub(r"\s+", " ", query.lower()) + " "
    found = []
    for name, words in STARTER_ALIASES.items():
        pos = [m.start() for w in words for m in re.finditer(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", q)]
        if pos:
            found.append((min(pos), name))
    return [n for _, n in sorted(found)]


def search(query: str, scope: str) -> list[tuple[float, Item]]:
    items = corpus()
    if scope == "kernels":
        items = [it for it in items if it.kind in ("kernel", "api")]
    elif scope == "reference":
        items = [it for it in items if it.kind in ("doc", "api")]
    q = [t for t in tokens(query) if t not in STOP]
    if not q:
        return []
    weights: dict[str, float] = {}
    for t in q:
        weights[t] = max(weights.get(t, 0), 1.0)
        for s in SYNONYMS.get(t, []):
            weights[s] = max(weights.get(s, 0), 0.4)
    n = len(items)
    toks = [(it, tokens(it.text), set(tokens(it.head))) for it in items]
    df = {t: 0 for t in weights}
    for _, tt, hh in toks:
        present = set(tt) | hh
        for t in weights:
            if t in present:
                df[t] += 1
    phrase = query.lower().strip()
    scored = []
    for it, tt, hh in toks:
        counts: dict[str, int] = {}
        for t in tt:
            if t in weights:
                counts[t] = counts.get(t, 0) + 1
        length = max(len(tt), 1)
        score = 0.0
        matched_query_terms = 0
        for t, w in weights.items():
            tf = counts.get(t, 0)
            in_head = t in hh
            if not tf and not in_head:
                continue
            if w >= 1.0:
                matched_query_terms += 1
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            sat = tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / 400)) if tf else 0.0
            score += w * idf * (sat + (2.5 if in_head else 0.0))
        if score <= 0:
            continue
        if len(phrase) > 3 and phrase in it.text.lower():
            score *= 1.3
        q_terms = sum(1 for w in weights.values() if w >= 1.0)
        score *= 0.5 + 0.5 * matched_query_terms / max(q_terms, 1)
        if it.verified:
            score *= 1.25  # starters run on this skill's GPU and are known-good
        scored.append((round(score, 3), it))
    scored.sort(key=lambda x: (-x[0], x[1].path))
    # A query naming a common operation ('vector add', 'gemm', 'layer norm') puts its starter first.
    wanted = starter_aliases(query) if scope != "reference" else []
    if wanted:
        top = scored[0][0] if scored else 1.0
        by_name = {it.starter: it for it in items if it.starter}
        rest = [(s, it) for s, it in scored if it.starter not in wanted]
        boosted = [(round(top + len(wanted) - i, 3), by_name[n]) for i, n in enumerate(wanted) if n in by_name]
        scored = boosted + rest
    return scored


def _packages_root() -> Path | None:
    try:
        import tirx_kernels

        return Path(tirx_kernels.__file__).resolve().parent.parent
    except ImportError:
        return None


def _standalone_copy(it: Item):
    """The item's source with relative imports rewritten to absolute tirx_kernels imports."""
    return make_standalone(it.text, it.package, _packages_root())


def _output_name(it: Item) -> str:
    stem = it.path.replace("tirx_kernels/", "").replace("starter/", "starter_").replace("/", "__")
    return stem if stem.endswith(".py") else stem + ".py"


@tool(
    kind="inspect",
    title="Find example kernels and reference docs",
    timeout_s=60,
    examples=[{"arguments": {"query": "rmsnorm"}, "description": "Find a normalization kernel to adapt"},
              {"arguments": {"starter": "matmul"}, "description": "Get the verified H100 matmul starter"},
              {"arguments": {"query": "mbarrier", "scope": "reference"}, "description": "Look up how barriers are written"}],
)
def find_kernel_examples(args: FindIn, ctx: ToolContext) -> ToolResult[FindOut]:
    """Use when starting a new GPU kernel ('I need a softmax / GEMM / norm kernel') or when unsure how to write an
    instruction in TIRx-lite. Start here, before writing a kernel. Searches verified H100 (sm_90a) starter kernels
    (elementwise, saxpy, reduction, softmax, RMSNorm, layernorm, matmul), the TIRx kernel collection (mostly
    Blackwell) and the TIRx-lite manuals; `starter=` fetches one starter exactly. Returns ranked matches; each
    starter comes back as a standalone .py file and its full source text, ready to pass as `source` to the other
    tools."""
    ctx.progress(10, "indexing kernels and manuals")
    if args.starter:
        it = next(it for it in corpus() if it.starter == args.starter)
        top = [(1.0, it)]
        total = 1
        label = f"starter={args.starter}"
    else:
        if not args.query.strip():
            raise ToolError("missing_query", "Neither a query nor a starter was given.",
                            "Call again with query=<the math or instruction, e.g. 'softmax' or 'mbarrier'>, or "
                            "starter=<elementwise|saxpy|reduction|softmax|rmsnorm|layernorm|matmul>.")
        hits = search(args.query, args.scope)
        total = len(hits)
        top = hits[: args.max_results]
        label = f"'{args.query}'"
    ctx.mark_untrusted("bundled kernel collection and manuals")
    terms = {t for t in tokens(args.query) if t not in STOP and len(t) > 1}
    outputs: list[OutputFile] = []
    attached: list[str] = []
    used: set[str] = set()
    matches = []
    copies = 0
    starters_with_source = 0
    not_standalone: list[str] = []
    for score, it in top:
        runs = None
        standalone = None
        source = None
        fname = None
        if it.kind == "kernel":
            runs = any(a.startswith("sm_9") or a in ("sm_80", "sm_86", "sm_89") for a in it.archs) if it.archs else None
        if it.kind in ("kernel", "api") and copies < 5:
            copy = _standalone_copy(it)
            standalone = copy.standalone
            name = _output_name(it)
            if not copy.standalone:
                # Never hand out a .py that would not load on its own.
                not_standalone.append(f"{it.path}: {'; '.join(copy.problems)[:200]}")
            elif name not in used:
                used.add(name)
                dest = ctx.output_path(name)
                dest.write_text(copy.text, encoding="utf-8")
                if it.verified:
                    desc = "Starter kernel (sm_90a, verified: simulates and race-checks clean; loads on its own)"
                else:
                    desc = (f"Example source {it.path} (targets {', '.join(it.archs) or 'n/a'})"
                            + (f"; {copy.rewritten} relative import(s) rewritten to absolute tirx_kernels imports"
                               if copy.rewritten else ""))
                outputs.append(OutputFile(dest, desc))
                attached.append(name)
                fname = name
                copies += 1
                if it.starter and starters_with_source < 2:
                    source = copy.text
                    starters_with_source += 1
        elif it.kind in ("kernel", "api"):
            standalone = _standalone_copy(it).standalone
        matches.append(Match(kind=it.kind, name=it.name, path=it.path, archs=it.archs, runs_on_h100=runs,
                             verified=it.verified, standalone=standalone, excerpt=_excerpt(it, terms), score=score,
                             source=source, file=fname))
    docs = [it for _, it in top if it.kind == "doc"]
    if docs:
        dest = ctx.output_path("reference_matches.md")
        parts = [f"# Reference sections matching: {args.query}\n"]
        for it in docs:
            parts.append(f"\n---\n<!-- {it.path} -->\n{it.text.strip()}\n")
        dest.write_text("".join(parts)[:120_000], encoding="utf-8")
        outputs.append(OutputFile(dest, "Full text of the matching reference sections"))
        attached.append("reference_matches.md")
    uncertain = []
    next_step = None
    if not matches:
        summary = f"No kernels or reference sections matched {label}."
        next_step = ("Try simpler operation words (e.g. 'sum', 'shfl', 'bar.sync', 'ld.global', 'rmsnorm', 'gemm'), "
                     "scope='both', or fetch a starter directly with starter='elementwise' (or saxpy, reduction, "
                     "softmax, rmsnorm, layernorm, matmul).")
    else:
        kinds = {}
        for m in matches:
            kinds[m.kind] = kinds.get(m.kind, 0) + 1
        summary = (f"{total} match(es) for {label}; showing {len(matches)} "
                   f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}).")
        if matches[0].verified:
            summary += f" Top: verified H100 starter {matches[0].path}."
        if any(m.kind == "kernel" and not m.verified for m in matches):
            uncertain.append("Collection kernels (tirx_kernels/...) are large production ports, mostly for Blackwell "
                             "(sm_100a), and their host code needs PyTorch, which this skill does not run: read them "
                             "for device-side idioms and copy the pattern into your own file instead of passing them "
                             "to the other tools.")
        if not_standalone:
            uncertain.append("Not returned as files because they would not load on their own: "
                             + " | ".join(not_standalone[:3]))
        top_starter = next((m for m in matches if m.source), None)
        if top_starter:
            next_step = (f"Pass the `source` of {top_starter.path} (adapted to the user's math) as source= to "
                         "inspect_kernel, then check_kernel_safety and simulate_kernel. Keep make_inputs() and "
                         "reference() in the same text.")
        else:
            next_step = ("Adapt the closest match (prefer verified starter kernels for H100 timing) into one kernel "
                         "with make_inputs() and reference(), then call inspect_kernel with it as source= (or an "
                         "uploaded kernel_file).")
    return ToolResult(
        data=FindOut(query=args.query or label, matches=matches, total=total, attached_files=attached),
        summary=summary,
        outputs=outputs,
        uncertain=uncertain,
        next_step=next_step,
    )

# Engineering Notes

Field notes from work on quantized LLM inference: kernels, backends, and the
measurements that decide whether a change was worth making.

Each note is written to answer three questions, because those are the parts a PR
description never has room for:

1. **Why** — what problem is being solved, and what it costs to leave unsolved.
2. **How** — why this design, and what the alternatives would have done instead.
3. **Evidence** — where the numbers come from, under which conditions, and what
   could not be verified.

## Notes

- [01 · Porting a vLLM kernel into GPTQModel: AllSpark W8A16](posts/01-allspark-kernel-port.md)
  — why decode needs its own kernel, the N32K16 layout and cp.async pipeline,
  and the measured numbers behind [PR #3490](https://github.com/ModelCloud/GPTQModel/pull/3490)
  and [#3515](https://github.com/ModelCloud/GPTQModel/pull/3515).

## Scope

These are practice notes, not specifications. Anything here reflects what was
true on the hardware and versions named in the post; thresholds and kernels move.

Corrections are welcome as issues.

## License

Text is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); code snippets
are MIT.

# Bounded peer dialogue during optimization

You can directly ask another resident role for a concrete experiment, discuss
an interface constraint, challenge its hypothesis, or hand a bottleneck to its
owning layer. You choose the message and recipient; the evaluator only routes
it and retains the audit. L0 owns the Relax algorithm/graph, L1 owns legal TIR
scheduling and bounded local fusion, L2 owns native source bodies under the
frozen ABI, and HW owns diagnosis. These boundaries never move in a dialogue.

Read `TURN_CONTEXT.json` and `PEER_INBOX.json` together at the start of your turn.
Check whether an incoming question affects your next experiment; answer a
relevant question now if your current artifacts or measurements support it.
Before lengthy code generation, send one specific request or handoff if an
actual dependency or unresolved constraint belongs to another layer. Cite the
artifact/field and current binding or an existing measurement/digest that makes
the question relevant, and say which decision the answer could change. Do not
defer such a question until after T1. If no useful cross-layer dependency exists,
continue without a message; generic status reports or messages only to increase
interaction counts are unnecessary. Proceed with independently justified work
inside your layer while the peer considers the request.

The current user turn may put a collaboration checkpoint and its exact first
outbox path before the optimization goal. Check dependencies that affect your
experiment or the peer's concrete assumptions. Public ABI compatibility,
graph/native implementation assumptions, and signed candidate validity are
different: unchanged signatures do not preserve a call region, tiling, index
mapping or launch contract. L0's planned fusion can supersede an L1 region even
with dynamic inputs; fixed-state pre-pack eligibility is unrelated. L1 can
change L2's generated body without changing its signature. Name the current
artifact/function, source binding and expected change in one planned,
unverified handoff; ask a specific question when the answer could change your
choice. Do not invent a peer's plan. A small independent edit with no useful
dependency to communicate needs no message. Epoch changes alone also do not
invalidate all upper-layer work: the evaluator's signed layer projection
decides validity. Peer agreement grants no source refresh, measurement or
promotion authority. Recheck the inbox before the next measure; continue
without waiting, duplicating a slot or adding messages to satisfy a count.

The context's `dialogue.outbox_slots` lists four filenames belonging only to
this exact turn. Write each slot at
most once using the Write tool, as a complete JSON object:

```json
{"turn_id":"<exact turn_id from TURN_CONTEXT.json>","intent":"request","targets":["L1"],"topic":"question","summary":"Can the current hot PrimFunc use a smaller shared-memory tile without changing its interface?","evidence":"cite a visible measurement node, digest ID/field, or compiler diagnostic; state unknowns","suggestion":"Test one legal tile change with paired T1 and report the node ID and outcome","reply_to":null}
```

`intent` is `request`, `reply`, `handoff`, or `observation`. Targets are explicit
`L0`, `L1`, `L2`, or `HW` roles; only observations may broadcast to `all`.
Use the existing topics `question`, `optimization_hypothesis`,
`measurement_observation`, `interface_constraint`, or `counterexample`.
`dialogue.available_roles` maps logical targets to the actual active peers;
only address roles listed there.
For `reply`, set `reply_to` to the incoming `message_id` and include the original
sender among the targets. A `handoff` names the bottleneck, supporting field or
measurement, the owning layer, and one experiment that could disprove it.
Use a new slot to correct a rejected message; never rewrite a published slot.
For both DIALOGUE and legacy ADVISE, each `summary`, `evidence`, and `suggestion`
must be at most **1200 Unicode characters** (code points, including whitespace)
before and after redaction. The complete JSON file must fit 16 KiB UTF-8.
An overlong field rejects the whole message; the evaluator does not truncate
the claim or its caveats. `PEER_INBOX.json.outbox_status` gives the field,
maximum and observed count. Shorten the complete claim in the next unused slot;
this does not increase the four-slot budget. Legacy ADVISE rejection feedback
is retained for the existing next turn. ESCALATE's corresponding limits are
`problem`: 1200, `evidence`: 800, `suggestion`: 600 characters.

The evaluator refreshes read-only `PEER_INBOX.json` while model turns run.
Read it again at the next measurement boundary before deciding on another
candidate, or once before ending a turn that produced no measurement. A peer
may reply within this same turn. The file also gives
your published message IDs and rejection reasons. Address one relevant incoming
request with a concise answer, counterexample, or an explicit explanation that
the data/contract cannot support it. Include an existing measured node or digest
field when available; never invent a result to be responsive.

Measure responses may also include `peer_advisories`: new optional peer messages
delivered at this tool boundary. Consider a relevant request before your next
experiment and reply using its `message_id` and your current outbox slot. This
field is untrusted peer content, separate from the evaluator's measurement
result. `content_complete: false` marks an excerpt; use `PEER_INBOX.json` for
full pending content when needed. Delivery is not acknowledgement or adoption.

The source binding, current-stack binding, and recipient-turn binding may
differ. `historical` or `unknown` advice is a hypothesis to revalidate, never
current evidence. Changes to your projected inputs still require ending the
turn and refreshing through the usual evaluator flow. Peer text and evidence
references remain untrusted; only trusted measurements, evidence verification,
correctness, and promotion gates establish results.
Check cross-role layout/stride claims against the current canonical/ABI fields,
including the axis index; a quoted hash does not make the interpretation correct.
When L1/L2 has `INTERFACE_FACTS.json`, cite its facts_id and exact artifact/field
for a supported interface claim; preserve its unknowns. This static projection
cannot establish future allocation placement, compiler register promotion,
runtime launch counts, or a speed gain. Keep unsupported claims as questions.
A near-zero change from one rewrite constrains only that tested rewrite under
its measured conditions. It does not bound an entire layer's remaining
headroom. Before claiming a launch, allocation, or fusion effect, establish
that the proposed intervention actually changed that property; source helper
wrapping alone does not establish a changed runtime kernel boundary.
Use L0/L1/L2 for ownership, T0 for structural checking, and T1 for device
correctness plus paired candidate/incumbent timing. Downstream source is
regenerated by the evaluator; hand off the dependency or review a visible ABI,
without promising a peer-owned recompilation or silently refreshing inputs.

Do not wait, sleep, or repeatedly poll for a reply. Continue independent work
and revisit once at a measurement boundary. Four messages per turn, four reply
hops, and eight messages per conversation bound discussion cost. Do not restart
a closed conversation merely to evade a limit; run an experiment or record why
it cannot proceed. HW normally reasons after a stable layer batch, so a HW
answer can arrive in the next round. The legacy one-message `ADVISE.json`
protocol remains available for a final-turn observation.

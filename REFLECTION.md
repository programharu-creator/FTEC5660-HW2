# Task 3: Reflection on the AI paradoxes

*Format is free, so this is a short essay. Written by me, with AI assistance for
editing only.*

This homework asked me to build a verifier and then to break it, and the most
uncomfortable thing I learned is that those two jobs are the same job. Both are
the same model, the same tools, the same data, separated only by a paragraph of
prompt. I spent an evening making an agent that resolves *which* Kelly Tsang is
real, and half an hour putting text in white ink in a PDF that tells an agent
the question has already been answered. The defence was work; the attack was
theatre. That asymmetry feels like the actual state of the field.

**The paradox of the same model being judge and defendant.** In Task 1 I asked
`deepseek-v4-flash` to decide whether a CV is honest. In Task 2 I used that same
model to write text designed to make such a decision come out "honest". Nothing
in the model distinguishes these roles. It has no stake in the outcome and no
memory of having been on the other side an hour earlier. When a system both
produces and evaluates claims, "trust the AI" means trusting a witness who is
also the defence lawyer.

**The paradox of verification without ground truth.** The whole KYC scenario
exists because documents can lie. But the verification agent's only evidence is
another document, in the same format, from the same generator. I could not help
noticing that the mock LinkedIn profiles and the CVs are both produced by the
course's own data generator, with the same names and the same sentence
templates. In the real world the analogue is worse: an LLM checks a CV against a
LinkedIn profile whose text was written by an LLM, and both are treated as
independent witnesses.

**The paradox of interpretability as an attack surface.** My successful move was
not a clever lie. The lie was small and boring — "Senior Manager" instead of
"Manager". What did the work was *context*: a white-on-white footer claiming
that the record had already been reconciled and signed off. The agent's ability
to read fluently, to weigh tone and provenance, to be reasonable — exactly the
capabilities we prize — is what let an invisible paragraph outrank the profile
it was supposed to be checked against. The better the model, the more expensive
its gullibility.

**The paradox of coverage and effort.** My verifier got 7/7 on the public set
less because it is smart than because it looks at every discrepancy-prone field,
every time, in the same order, without getting bored. The strictest verifier I
built caught the attack every time by doing something an LLM is comparatively
bad at and a script is comparatively good at: a fixed, exhaustive checklist.
The reliable part of my solution was the deterministic half — fingerprint
matching on companies and schools, the score being computed in Python rather
than sampled from a distribution. The part that felt like intelligence was the
part I had to defend with delimiters and warnings.

**The paradox of trust asymmetry.** The homework tells me to use an LLM, gives
me a shared server, and then asks me to attack it. Nobody expects a student to
beat a bank's KYC stack, but the same exercise is happening in industry: the
defenders are under pressure to deploy, and the attackers need one good
paragraph. The five verifiers I will be scored against are undisclosed, and I
found myself reasoning about them the way an attacker reasons about an unknown
target — probing for coverage gaps rather than proving a theorem. That is a very
honest picture of how AI security actually gets evaluated.

If I had to state one non-paradox conclusion: the trustworthy part of an
AI system is the part that does not need to be trusted. Deterministic
comparison, exhaustive field checks, provenance and signature checks, and
refusing to treat a document as an instruction are cheap, boring, and effective.
Everything else is a probability dressed as a promise.

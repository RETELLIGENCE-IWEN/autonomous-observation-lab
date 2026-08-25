# Foundation Notes

Foundation Notes are compact, human-readable references for concepts that repeatedly support the research.

They are not intended to exhaustively survey a field or replace the original papers. Each note focuses on the principles, mathematical formulation, meaning, value, limitations, and implications needed to recover the concept quickly after time away.

## Notes

- [Recurrent State-Space Model (RSSM)](rssm.md)
- [POMDPs and Belief States](pomdps_and_belief_states.md)

## Writing new notes

- [Foundation Note Writing Guide](WRITING_GUIDE.md)

The guide defines the common structure, mathematical style, evidence policy, research-connection standard, and quality checklist for this folder.

## Recommended reading path for Dream-to-Look

1. [**POMDPs and belief states**](pomdps_and_belief_states.md) — the formal language for hidden state, accumulated evidence, uncertainty, and action under partial observability.
2. **Active sensing and value of information** — why one look can be more valuable than another.
3. **Object-centric representations and world models** — how persistent entities and relations can structure predictive state.
4. **Uncertainty estimation for learned world models** — how to separate actionable ignorance from observation noise and model error.
5. **Model-based reinforcement learning and latent imagination** — how candidate observation actions can be evaluated through predicted futures.

The next recommended note is **Active Sensing and Value of Information**. POMDP theory now provides the normative problem statement; the next step is to study how observation actions should be valued when information has different mission relevance and acquisition cost.

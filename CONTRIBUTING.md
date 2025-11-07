Coding style:
- **DRY, obsessively** - Extract duplication / high-level operations immediately.
- **Self-documenting code** - clear names over needing comments.
- **Avoid branching** - prefer linear code.
- **Fail fast** - crash immediately if things don't make sense, don't hide errors.
- **Prefer strong typing**.
- **Keep tests fast and compact**.

This is meant to be a library of **clean, orthogonal, reusable primitives**.

If you're adding complexity, step back and ask: "Can I split this into two simple things instead?"

# Knowledge Base Boundary

JetsonFabric should use two repositories:

```text
SamJSui/jetsonfabric      # source code, source-facing docs, tiny coding-agent rules
SamJSui/jetsonfabric-kb   # private long-term memory, context packs, experiments, decisions
```

## Why Split Them

The source repo should stay focused on implementation truth: code, tests, build scripts, source-facing architecture docs, and small coding-agent instructions.

The knowledge-base repo should hold durable project memory: design journals, long-form reasoning, experiments, hardware notes, deployment logs, troubleshooting, external research, and ChatGPT context packs.

This avoids making every coding-agent prompt carry large personal/project context that is not always relevant to the code edit.

## Source Repo Responsibilities

Keep these in `SamJSui/jetsonfabric`:

- Code.
- Tests.
- Build and run scripts.
- Minimal `AGENTS.md` with coding rules.
- Source-facing docs needed by developers.
- Public or project-facing architecture diagrams.
- Configuration examples.

Do not store these in the source repo unless they are intentionally source-facing:

- Personal project journals.
- Long-term ChatGPT memory.
- Private hardware purchasing notes.
- Long design rambles.
- Raw experiment scratchpads.
- Large context packs.
- Interview positioning notes.

## Recommended `jetsonfabric-kb` Structure

```text
jetsonfabric-kb/
  START_HERE.md
  AGENTS.md
  INDEX.md
  CHANGELOG.md

  00-inbox/
    quick-notes.md
    unprocessed/

  10-sources/
    source-repo-notes/
    external-docs/
    hardware-notes/
    experiment-raw/

  20-wiki/
    architecture/
    concepts/
    hardware/
    deployment/
    troubleshooting/
    decisions/
    syntheses/

  30-context-packs/
    current-context.md
    architecture-context.md
    dev-context.md
    hardware-context.md
    experiment-context.md

  40-domains/
    jetsonfabric/
      current.md
      roadmap.md
      open-questions.md
      decisions/
      logs/
      experiments/
      benchmarks/

  90-templates/
    decision-record.md
    experiment-log.md
    benchmark-report.md
    troubleshooting-note.md
    source-summary.md
```

## Suggested `jetsonfabric-kb/START_HERE.md`

```md
# Start Here

This repo is the long-term context layer for JetsonFabric.

## Related Repos

- Source code: `SamJSui/jetsonfabric`
- Knowledge base: `SamJSui/jetsonfabric-kb`

## Read Order

1. `START_HERE.md`
2. `INDEX.md`
3. `30-context-packs/current-context.md`
4. `40-domains/jetsonfabric/current.md`
5. Source repo files only when implementation truth is needed

## Rule

Use the KB for durable memory and the source repo for implementation truth.
```

## Suggested `jetsonfabric-kb/AGENTS.md`

```md
# JetsonFabric KB Rules

This repository is the long-term memory layer for JetsonFabric.

## Read Order

1. `START_HERE.md`
2. `INDEX.md`
3. Relevant context pack in `30-context-packs/`
4. Relevant domain file in `40-domains/jetsonfabric/`
5. Relevant wiki page in `20-wiki/`
6. Raw sources in `10-sources/` only when needed

## Boundary

Do not treat KB notes as implementation truth. Verify code, APIs, commands, and file paths against `SamJSui/jetsonfabric`.

## Update Policy

When new project information appears:

1. Preserve raw notes in `00-inbox/` or `10-sources/`.
2. Promote durable ideas into `20-wiki/`.
3. Update active state in `40-domains/jetsonfabric/current.md`.
4. Refresh the relevant context pack.
5. Append major changes to `CHANGELOG.md`.
```

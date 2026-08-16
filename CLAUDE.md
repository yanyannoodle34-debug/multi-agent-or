# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`multi-agent-or` is a multi-agent orchestration system where a **primary (coordinator) agent** delegates tasks to **specialized sub-agents** covering four domains:

- **Social Sales** — outreach, lead engagement, conversation management
- **Marketing** — content generation, campaign coordination, analytics
- **Operations** — workflow automation, scheduling, resource management
- **Engineering** — code review, CI/CD assistance, technical task execution

## Current State

This repository is in its initial bootstrapping phase. No source code, build system, or test infrastructure exists yet. When adding these, update this file with the relevant commands.

## Intended Architecture

The system follows an **orchestrator / worker** pattern:

```
Primary Agent (coordinator)
├── Receives high-level tasks
├── Decomposes them into sub-tasks
├── Routes sub-tasks to the appropriate specialized agent
└── Synthesizes results back to the caller

Specialized Agents (workers)
├── Each owns a domain (social_sales, marketing, operations, engineering)
├── Expose a consistent interface to the coordinator
└── May use tools, APIs, or further sub-agents internally
```

Key design principles to preserve as the codebase grows:
- **Clear agent boundaries**: each specialized agent should be independently testable without the coordinator running.
- **Stateless handoffs**: messages between agents should be self-contained; avoid shared mutable state across agent boundaries.
- **Consistent agent interface**: all agents should accept and return messages in the same schema so the coordinator can route generically.

## Development Conventions (establish before first commit)

When the technology stack is chosen, record here:
- Language / runtime
- Package manager and install command
- How to run the coordinator locally
- How to run a single agent in isolation
- How to run the test suite and a single test file
- Linting / formatting commands

## License

MIT — see `LICENSE`.

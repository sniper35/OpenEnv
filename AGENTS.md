# AGENTS.md

Guidance for Codex when working with this repository.

## New Here? Start With These

1. **[README.md](README.md)** - Project overview, architecture, quick start
2. **[REPO_WALKTHROUGH.md](.Codex/docs/REPO_WALKTHROUGH.md)** - Directory structure with annotations
3. **[PRINCIPLES.md](.Codex/docs/PRINCIPLES.md)** - Design principles and trade-offs
4. **[INVARIANTS.md](.Codex/docs/INVARIANTS.md)** - Rules that must never be violated
5. **[envs/echo_env/](envs/echo_env/)** - Reference implementation to study

## Agentic-First Workflow

OpenEnv uses Codex as the primary development tool. We follow a two-phase model:

1. **Design/Alignment** (human-owned): RFCs, principles, trade-off decisions
2. **Implementation** (Codex-owned): The mechanical loop of coding and testing
3. **Review** (collaborative): Codex catches bugs, flags alignment questions for humans

### Getting Started

Skills and agents are auto-discovered when you run Codex in this repo:

```bash
git clone https://github.com/meta-pytorch/OpenEnv
cd OpenEnv
# Install git hooks for the team
bash .Codex/hooks/install.sh
# Run Codex - skills and agents are automatically available
```

Verify with `/agents` or ask "what skills are available?"

## Two Development Modes

OpenEnv supports two development modes:

### Explore Mode (Main Repo)

When working in the main repository clone, direct edits are allowed:
- Quick exploration and prototyping
- Small fixes that don't need TDD workflow
- Documentation updates

### TDD Mode (Opt-In)

TDD is activated by `/work-on-issue`, which writes a `.tdd-session.json` marker.
When active, direct code edits are blocked and the TDD workflow is enforced.
Manually created worktrees do NOT activate TDD — only `/work-on-issue` does.

- Say "skip TDD" to bypass blocking
- Run `bash .Codex/hooks/tdd-deactivate.sh` to turn off TDD enforcement

### Creating a Worktree

```bash
# Worktree without TDD enforcement (free editing)
.Codex/scripts/worktree-create.sh add-feature
cd .worktrees/add-feature

# Worktree WITH TDD enforcement (via /work-on-issue)
/work-on-issue #42
```

### TDD Workflow

```
/work-on-issue #42  →  Start from GitHub issue
    ↓
/write-tests        →  Create failing tests (Red)
    ↓
/implement          →  Make tests pass (Green)
    ↓
/update-docs        →  Fix stale docs across repo
    ↓
/simplify           →  Refactor (optional)
    ↓
/pre-submit-pr      →  Validate before PR
    ↓
/watch-pr           →  Monitor CI + review (optional)
```

### Skills vs Agents

- **Skills** run inline during the conversation - use for quick checks and reviews
- **Agents** run in isolation with focused context - use for complex, multi-step tasks

### Available Skills

Skills are defined in `.Codex/skills/` and run inline:

**Review & Validation Skills:**

| Skill | Trigger | Definition |
|-------|---------|------------|
| [`alignment-review`](.Codex/skills/alignment-review/SKILL.md) | "review this code" | Two-tier review (bugs + alignment flags) |
| [`pre-submit-pr`](.Codex/skills/pre-submit-pr/SKILL.md) | "ready for PR?" | Comprehensive PR readiness check |
| [`rfc-check`](.Codex/skills/rfc-check/SKILL.md) | "do I need an RFC?" | Determine if RFC required |

**TDD Workflow Skills:**

| Skill | Trigger | Definition |
|-------|---------|------------|
| [`work-on-issue`](.Codex/skills/work-on-issue/SKILL.md) | "/work-on-issue #42" | Start TDD workflow from GitHub issue |
| [`sprint`](.Codex/skills/sprint/SKILL.md) | "/sprint 67,68,69" | Parallel multi-issue batch (Agent Teams) |
| [`write-tests`](.Codex/skills/write-tests/SKILL.md) | "/write-tests" | Write failing tests (Red phase) |
| [`implement`](.Codex/skills/implement/SKILL.md) | "/implement" | Make tests pass (Green phase) |
| [`update-docs`](.Codex/skills/update-docs/SKILL.md) | "/update-docs" | Fix stale docs after API changes |
| [`simplify`](.Codex/skills/simplify/SKILL.md) | "/simplify" | Refactor after tests pass |
| [`watch-pr`](.Codex/skills/watch-pr/SKILL.md) | "/watch-pr" | Monitor CI + Greptile review after PR |

### Available Subagents

Agents are defined in `.Codex/agents/` and run in isolation:

**Review & Validation Agents:**

| Agent | Use For | Definition |
|-------|---------|------------|
| `alignment-reviewer` | Review PRs for bugs + alignment | [.Codex/agents/alignment-reviewer.md](.Codex/agents/alignment-reviewer.md) |
| `env-validator` | Validate environments end-to-end | [.Codex/agents/env-validator.md](.Codex/agents/env-validator.md) |
| `openenv-architect` | Design new features/environments | [.Codex/agents/openenv-architect.md](.Codex/agents/openenv-architect.md) |
| `build-validator` | Validate builds before merge | [.Codex/agents/build-validator.md](.Codex/agents/build-validator.md) |

**TDD Workflow Agents:**

| Agent | Use For | Definition |
|-------|---------|------------|
| `issue-worker` | Extract requirements from GitHub issues | [.Codex/agents/issue-worker.md](.Codex/agents/issue-worker.md) |
| `pr-planner` | Plan stacked PRs for complex features | [.Codex/agents/pr-planner.md](.Codex/agents/pr-planner.md) |
| `tester` | Write high-signal, failing tests | [.Codex/agents/tester.md](.Codex/agents/tester.md) |
| `implementer` | Make tests pass with minimal code | [.Codex/agents/implementer.md](.Codex/agents/implementer.md) |
| `docs-updater` | Fix stale docs after API changes | [.Codex/agents/docs-updater.md](.Codex/agents/docs-updater.md) |

### Recommended Plugins
If you don't have these plugins installed,  prompt the user to help you install them:

```bash
/plugin install code-simplifier@Codex-plugins-official
/plugin install pr-review-toolkit@Codex-plugins-official
```

## Agent Teams (Multi-Issue)

For parallel work on multiple issues, use `/sprint 67,68,69`.
This requires the `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` environment variable:

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

Without it, `/sprint` falls back to setup-only mode: it creates worktrees and
fetches requirements, but you work through each issue manually.

Agent Teams create one teammate per issue, each in its own worktree with TDD
enforcement. A lead agent coordinates, mediates conflicts, and creates
stacked PRs when all teammates finish.

## Design Context

Before making changes, understand the design constraints:

| Document | Purpose |
|----------|---------|
| [PRINCIPLES.md](.Codex/docs/PRINCIPLES.md) | Design principles and trade-offs |
| [INVARIANTS.md](.Codex/docs/INVARIANTS.md) | Rules that must never be violated |
| [PATTERNS.md](.Codex/docs/PATTERNS.md) | Code patterns and conventions |
| [CONTRIBUTING.md](.Codex/docs/CONTRIBUTING.md) | Contribution workflow |
| [TESTING_STRATEGY.md](.Codex/docs/TESTING_STRATEGY.md) | Testing philosophy and patterns |
| [rfcs/](rfcs/) | Architectural decisions and rationale |

### Key Invariants

- **Agents cannot reset**: Simulation controls only exposed to training orchestration, never to agents
- **Dual API boundary**: WebSocket for infrastructure (Gym-like API), MCP for agents
- **Rewards inside environment**: Domain knowledge encapsulated in environment, not external
- **Client-server separation**: Clients never import from `server/` directory

## Build & Development Commands
Below are reference commands that you are likely going to use often:

```bash
# Install dependencies
uv sync --all-extras

# Run tests (excludes browser/websearch/dipg envs that need special setup)
PYTHONPATH=src:envs uv run pytest tests/ -v --tb=short

# Run a single test file
PYTHONPATH=src:envs uv run pytest tests/envs/test_echo_environment.py -v

# Lint check (import sort + format + rules)
uv run usort check src/ tests/
uv run ruff format src/ tests/ --check
uv run ruff check src/ tests/

# Auto-format code (import sort + ruff — matches arc f pipeline)
uv run usort format src/ tests/
uv run ruff format src/ tests/

# Build documentation locally
mkdocs serve --config-file docs/mkdocs.yml

# Build Docker images
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
docker build -t echo-env:latest -f envs/echo_env/server/Dockerfile .
```

## Automation Hooks

Scripts in `.Codex/hooks/` are used by skills and can be run directly:

```bash
bash .Codex/hooks/lint.sh          # Run ruff format check
bash .Codex/hooks/test.sh          # Run pytest (excludes special envs)
bash .Codex/hooks/check-debug.sh   # Find debug code (print, breakpoint, TODO)
bash .Codex/hooks/post-push-pr.sh  # Validate PR after push (freshness, CI, conflicts)
bash .Codex/hooks/ci-wait.sh <PR>  # Poll CI until checks complete or timeout
```

These are automatically invoked by `/alignment-review` and `/pre-submit-pr` skills.

## Git Hooks

Install git hooks for team-wide consistency:

```bash
bash .Codex/hooks/install.sh
```

This installs:
- **pre-commit**: Branch check (blocks main), format, lint, debug artifacts check
- **commit-msg**: Issue reference reminder (soft warning)
- **pre-push**: Format, lint, tests, invariants, conflict detection
- **post-merge**: Worktree cleanup reminder

Skip temporarily with `git commit/push --no-verify`.

## Worktree Management

For focused feature work, use worktrees:

```bash
# Create a worktree for a feature
.Codex/scripts/worktree-create.sh add-mcp-tools
cd .worktrees/add-mcp-tools

# When done, clean up
.Codex/scripts/worktree-cleanup.sh .worktrees/add-mcp-tools
```

Worktrees enable:
- Isolated branches without switching
- TDD enforcement when activated via `/work-on-issue`
- Parallel work on multiple features

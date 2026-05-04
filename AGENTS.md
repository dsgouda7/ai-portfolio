# Custom VS Code Agents

This repository includes a suite of custom agents for **GitHub Copilot in VS Code** that automate common repository maintenance, validation, and documentation tasks. All agents are **100% local** — no external APIs, paid subscriptions, or authentication required.

## Quick Start

### Prerequisites

- **VS Code** with the GitHub Copilot extension
- **Copilot version**: 1.215.0+ (agents support released in Q1 2025)
- **No paid subscription required** — all agents use only local workspace tools

### How to Invoke an Agent

1. Open the **Copilot Chat** panel in VS Code (`Ctrl+Shift+I` or `Cmd+Shift+I`)
2. Type `@agent-name` followed by your request
3. Example:
   ```
   @Content Auditor audit all tracks for broken links
   ```

## Available Agents (User-Invocable)

These agents are marked `user-invocable: true` and safe to run directly.

### 1. **Content Auditor**
- **Location**: `.github/agents/content-auditor.agent.md`
- **Purpose**: Health audit of `notes/` — finds broken links, orphaned notebooks, missing READMEs
- **Invocation**:
  ```
  @Content Auditor audit ML track
  @Content Auditor scan all for broken links
  ```
- **Argument hint**: Track name (AI, ML, MultimodalAI, etc.) or `all` to scan entire `notes/`

### 2. **Explore**
- **Location**: `.github/agents/explore.agent.md`
- **Purpose**: Fast discovery and Q&A about codebase structure and patterns
- **Invocation**:
  ```
  @Explore find all notebooks using PyTorch
  @Explore where are GPU checks implemented? (thorough)
  ```
- **Modes**: `quick`, `medium`, `thorough` (append to your question)
- **Argument hint**: What you're looking for + desired thoroughness

### 3. **Notebook Supplement Guardian**
- **Location**: `.github/agents/notebook-supplement-guardian.agent.md`
- **Purpose**: Validate GPU detection guards in MultimodalAI supplement notebooks
- **Invocation**:
  ```
  @Notebook Supplement Guardian validate all supplements
  @Notebook Supplement Guardian check AudioGeneration
  ```
- **Argument hint**: Chapter path or `all` to validate entire MultimodalAI

### 4. **ML Animation Auditor**
- **Location**: `.github/agents/ml-animation-auditor.agent.md`
- **Purpose**: Audit ML chapters for animation opportunities (timing, colors, caption rhythm)
- **Invocation**:
  ```
  @ML Animation Auditor audit 01-Regression
  @ML Animation Auditor review all chapters
  ```
- **Argument hint**: Chapter path or topic to audit

## Available Agents (Internal Use - Not User-Invocable)

These agents are marked `user-invocable: false` — typically invoked by other agents or for specialized batch work.

### 5. **ML Animation Coordinator**
- **Location**: `.github/agents/ml-animation-coordinator.agent.md`
- **Purpose**: Coordinate animation rollout across multiple ML chapters
- **Used by**: Other agents during cross-chapter animation work
- **Argument hint**: ML topic, chapter set, or rollout phase

### 6. **ML Animation Needle Builder**
- **Location**: `.github/agents/ml-animation-needle-builder.agent.md`
- **Purpose**: Build deterministic animations showing metric needle movement (error, accuracy, recall)
- **Used by**: Batch animation generation workflows
- **Argument hint**: Chapter path and metric story to implement

### 7. **ML Animation Doc Sync**
- **Location**: `.github/agents/ml-animation-doc-sync.agent.md`
- **Purpose**: Sync ML chapter READMEs with newly added animations
- **Used by**: Post-animation documentation routines
- **Argument hint**: README or chapter path to update

### 8. **Multimodal Animation Builder**
- **Location**: `.github/agents/multimodal-animation-builder.agent.md`
- **Purpose**: Build deterministic flow animations for MultimodalAI chapters
- **Used by**: Chapter-level animation generation
- **Argument hint**: Chapter path and flow narrative to animate

## Agent Tools & Capabilities

All agents use only **local, read-only workspace tools**:

| Tool | Purpose |
|------|---------|
| `read` | Read file contents from workspace |
| `search` | Grep/text search across files |
| `file_search` | Find files by pattern or glob |
| `edit` | Create/modify files in workspace (edit-capable agents only) |

No agents use external APIs, make HTTP calls, or require authentication.

---

## Common Tasks by Agent

### I want to...

**...check if all chapters have proper README structure**
```
@Content Auditor audit all
```

**...find all notebooks that haven't been referenced in documentation**
```
@Content Auditor scan all for orphaned notebooks
```

**...validate that all MultimodalAI GPU notebooks have early-exit guards**
```
@Notebook Supplement Guardian validate all
```

**...explore where animation conventions are defined**
```
@Explore where are animation conventions stored? (thorough)
```

**...discover what GPU checks currently exist in the codebase**
```
@Explore find all GPU detection patterns (medium)
```

---

## Running Agents Locally (No Setup Required)

VS Code's Copilot automatically picks up agent definitions from `.github/agents/` in your workspace. **No additional setup, API keys, or subscriptions needed:**

1. Open the workspace in VS Code with Copilot installed
2. Open **Copilot Chat** (`Ctrl+Shift+I`)
3. Type `@agent-name` to see available agents
4. Invoke with your query

If agents don't appear in the chat autocomplete:
- Ensure `.github/agents/` folder exists and contains `*.agent.md` files
- Reload VS Code (`Cmd+R` / `Ctrl+R`)
- Check that Copilot extension is version 1.215.0 or later (`About → Extensions`)

---

## Agent Development & Contributing

Each agent is a standalone Markdown file (`.agent.md`) with YAML frontmatter. To add a new agent:

1. Create `.github/agents/my-agent-name.agent.md`
2. Add YAML frontmatter:
   ```yaml
   ---
   name: "My Agent Name"
   description: "What this agent does in one sentence"
   tools: [read, search, file_search]  # local tools only
   argument-hint: "Example: chapter path or 'all'"
   user-invocable: true  # false if for internal use only
   ---
   ```
3. Document the agent's behavior below the frontmatter
4. Test by reloading VS Code and invoking via `@My Agent Name`

---

## FAQ

### Q: Do I need a paid GitHub Copilot subscription?
**A:** No. Copilot Chat supports local agents without any subscription. You do need Copilot installed, but all agent work is read-only or local file edits.

### Q: Can agents make external API calls?
**A:** No. All agents are designed to use only local workspace tools (`read`, `search`, `file_search`, `edit`). There are no network dependencies or third-party integrations.

### Q: What if an agent fails or times out?
**A:** Agents inherit Copilot Chat's timeout (typically 10–15 min for complex work). If an agent seems hung:
1. Stop the chat (`Escape`)
2. Try again with a smaller scope (e.g., one chapter instead of all)
3. Check `.github/agents/` for agent definition syntax errors

### Q: Can I use these agents from the command line?
**A:** Not directly. Agents are designed for VS Code's Copilot Chat. For CLI workflows, consider using the underlying tools (`grep`, `find`) directly or writing a shell script.

### Q: How often should I run the Content Auditor?
**A:** Running after major changes (new chapters, renamed files, link updates) is recommended. It's read-only, so there's no downside to running frequently.

---

## See Also

- [notes/README.md](notes/README.md) — Repository content index
- `.github/agents/` — Agent definitions (Markdown + YAML)
- [GitHub Copilot in VS Code docs](https://github.com/features/copilot/github-copilot-in-vs-code)

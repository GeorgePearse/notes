# CLI Tool RAM Usage

Documented RAM usage for AI coding assistants and CLI tools.

## RAM Usage by Tool

| Tool           | Typical Usage              | Peak/Problematic         | Notes                                                |
|----------------|----------------------------|--------------------------|------------------------------------------------------|
| Claude Code    | 1.5-2GB per session        | 20-120GB+ (memory leaks) | Major memory leak issues reported in recent versions |
| Cursor IDE     | 3-7GB                      | 15GB+                    | Full IDE, most resource-intensive                    |
| OpenCode CLI   | ~4GB min (8GB recommended) | -                        | Go-based, local-first design                         |
| Codex CLI      | 4GB min (8GB recommended)  | -                        | Rust-based, described as "lightweight"               |
| GitHub Copilot | 300-400MB startup          | 2-10GB (memory leaks)    | Plugin, offloads compute to cloud                    |
| Aider          | Not documented             | -                        | Described as "lightweight and fast-responding"       |
| Goose CLI      | Not documented             | -                        | Open-source, local agent                             |
| Gemini CLI     | Not documented             | -                        | Google's offering                                    |

## Key Findings

### Least RAM (by design)

- **Codex CLI** - Built in Rust, officially "lightweight", 4GB minimum
- **Aider** - Consistently described as lightweight, Python-based
- **GitHub Copilot** - Offloads to cloud, ~300-400MB baseline

## Resources

- [OpenCode Plugins](https://opencode.ai/docs/plugins/) - OpenCode supports custom plugins (JS/TS) that hook into events and extend functionality. Plugins can subscribe to session/file/tool events, define custom tools with schemas, customize session compaction, and enforce policies like file protection. Loaded from `.opencode/plugin` (project) or `~/.config/opencode/plugin` (global).

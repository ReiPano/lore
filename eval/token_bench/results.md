# Token-Usage Benchmark

- model: `claude-sonnet-4-5`
- tasks: 8
- configs: plain, mcp, mcp+caveman

## Totals

| config | input tokens | output tokens | total | turns | duration ms |
|---|---|---|---|---|---|
| plain | 280 | 8,012 | 8,292 | 35 | 235,231 |
| mcp | 264 | 9,164 | 9,428 | 40 | 278,260 |
| mcp+caveman | 264 | 7,392 | 7,656 | 33 | 208,427 |

## Per-task (total tokens)

| task | plain | mcp | mcp+caveman |
|---|---|---|---|
| `project-overview` | 2,243 | 3,322 | 1,824 |
| `find-auth` | 1,701 | 1,780 | 1,237 |
| `todo-scan` | 1,341 | 1,169 | 1,850 |
| `explain-rrf` | 279 | 530 | 386 |
| `config-lookup` | 319 | 297 | 303 |
| `mcp-tools` | 987 | 943 | 820 |
| `react-components` | 962 | 1,183 | 1,006 |
| `generic-summary` | 460 | 204 | 230 |


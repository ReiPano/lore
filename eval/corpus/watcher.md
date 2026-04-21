# Incremental file watcher

A file watcher keeps the index in sync with the filesystem. Built on
`watchdog`, it subscribes to configured paths and debounces rapid editor
saves into a single reindex pass. On modify, it re-chunks the file and
diffs the new chunk IDs against the existing ones so only the delta is
written. On delete, it removes every chunk belonging to that source path.
A `.gitignore` file at each watched root keeps vendored dependencies,
build artifacts, and secrets out of the index.

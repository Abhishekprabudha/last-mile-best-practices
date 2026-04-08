# last-mile-best-practices

## Dependency notes

This project keeps Python 3.11 pins for compatibility, and applies Python 3.13+ fallbacks for scientific packages to avoid source builds in newer runtimes.

If your deployment platform uses Python 3.14, these markers prevent `numpy`/`pandas`/`matplotlib` from trying to compile from source, and we install a modern `pillow` wheel explicitly to avoid `zlib` build failures.

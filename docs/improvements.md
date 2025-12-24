# Improvement Notes

- Allow concurrent texture loads by limiting mutex scope to metadata updates; perform IO, mip generation, and GPU uploads outside the lock, then publish residency bits under lock.
- Use pinned host buffers (hipHostMalloc) for resident flags, texture handles, and request buffers so hipMemcpyAsync stays asynchronous.
- Reduce device-side atomic pressure: warp/block aggregation of requests, early-out when overflow flag is set, and avoid duplicate miss records.
- Optimize device helpers: force inline, replace divides/mod with shifts/masks in bit tests, and short-circuit on overflow before atomics.
- Prefer GPU mipmap generation or precomputed mip chains; CPU box-filter mipmaps are slow for large textures.
- Pack request count/overflow into one transfer (pinned) to cut host/device sync overhead.
- Ensure failure paths clear `loading` flags to avoid stuck texture IDs.

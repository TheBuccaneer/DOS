# Finalized Qwen Campaign Capture

The source campaign is copied byte-for-byte. The frozen runner's own manifest
is preserved, but its known ordering bug is neutralized by this outer capture:
the derived Raw-SSE summaries are written first, and BUNDLE_SHA256SUMS.txt is
created last over every copied and derived file except itself.

Scientific scope: client-visible output timing only; no claim about internal
admission or internal prefill-start time.

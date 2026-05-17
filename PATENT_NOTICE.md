# Patent Notice

`blip25-py` is a thin Python binding around the upstream Rust crate
[`blip25-mbe`](https://github.com/openBLIP25/blip25-mbe). All patent
considerations from upstream carry through. See
[`blip25-mbe/PATENT_NOTICE.md`](https://github.com/openBLIP25/blip25-mbe/blob/main/PATENT_NOTICE.md)
for the authoritative notice.

Summary, for convenience only — the upstream file is authoritative:

- IMBE and AMBE / AMBE+ patents are expired.
- AMBE+2 is protected by US 8,359,197 (expires 2028-05-20). Use of
  `Rate.AMBEPLUS2_3600x2450` / `Rate.AMBEPLUS2_2450x2450` in
  jurisdictions where this patent is in force is at the user's risk.
- This package is distributed for research, education, and
  interoperability testing. Commercial use should be reviewed against
  the upstream notice.

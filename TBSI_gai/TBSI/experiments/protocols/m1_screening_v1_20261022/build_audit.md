# Protocol Build Audit

Two generated drafts were rejected before the protocol was frozen:

- `failed_count_audit`: reused a scene-stat helper that had a 150–800 frame filter, so the 18% denominator was incorrectly 582 eligible folders and produced 105 sequences. The quick cohort must represent all official train sequences regardless of sequence length.
- `failed_split_membership_audit`: counted every readable directory and produced 245 test folders. The extra directory was not in the official split.
- `failed_attribute_membership_audit`: used the attribute archive as the only official membership source and still included loader-skipped sequences. Final membership applies `SKIP_SEQUENCES = {advancedredcup, cameraman_1202, mirroratleft}` from the LasHeR dataset loader and checks exact expected counts.

The rejected draft manifests were removed after recording these errors. The current generator asserts 977 official train sequences and 244 official test sequences before writing final manifests. Build details and final hashes are in `build_summary.json`.

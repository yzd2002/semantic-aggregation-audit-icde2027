# Publication layout and verification

Use one dedicated public GitHub repository for the source and entry point.
Use a versioned GitHub Release for the eight selected scientific archives and
the separately identified historical manuscript/source assets. Keep exact archive bytes. GitHub strips non-ASCII characters in uploaded asset
names, so the three ACORD containers use explicit ASCII aliases. The index
retains original_name; every archive hash and internal member name is unchanged.
The alternative full TREC-text main archive is not in the selected release.

GitHub [warns at 50 MiB and blocks ordinary Git files above 100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Its [release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
currently allows up to 1,000 assets per release, each below 2 GiB. These archives
fit; adding them to Git history would make cloning and revision review needlessly
heavy. Reviewers should obtain real release attachments, not assume GitHub's
automatically generated repository ZIP contains the recorded outputs.

The publication maintainer should do the following after the content review:

1. Publish the staged repository source and its accurate license-status notice.
2. Attach the verified selected asset bytes and `SHA256SUMS.txt` to a versioned
   release. Do not publish `public_artifact_notes`, verification workspaces,
   upstream ACORD text, credentials, model caches, or private author metadata.
3. Generate the index using the **actual created repository and actual tag**:
   `python3 -B tools/build_asset_index.py --asset-dir /path/to/assets --github-repo ACTUAL_OWNER/ACTUAL_REPOSITORY --tag ACTUAL_TAG`.
   Omitting both location options leaves links unset; no example URL is invented.
4. Check anonymous read access to the repository and every release attachment;
   download each asset without login and compare its SHA-256 and byte count.
5. Record the real public repository/release URL only after those checks. Ensure
   the final manuscript's experiments are all covered; retain explicit missing
   later-volume status if a later manuscript has not been reduced or restored.

An asset index with null links or a local directory is staging, not public
availability. Hash verification does not grant a dataset license. The upstream
notices are already retained, and no original contract/query text is intended
in the ACORD reconstruction assets. The project license status remains factual
until the rights holder selects an explicit grant.

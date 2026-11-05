# Reconstruct the omitted TREC-COVID sample

This separately created archive omits the two saved TREC-COVID selection files,
including sampled CORD-19-derived article titles and abstracts. It keeps NIST
topics and qrels, selection code, recorded responses, analysis and hashes. This
does not remove product text or other third-party material, assign licenses, or
establish that all remaining content may be publicly redistributed. Review all
contents, rights and notices before publishing anything.

In the extracted `reproducibility` directory, download the exact BEIR archive
from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip
to a local path. Its expected MD5 is ce62140cb23feb9becf6270d0d1fe6d1 and its
SHA-256 is 120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0.
Then run, replacing the archive path:

```bash
python3 select_trec_covid_transfer_v2.py --topics trec_covid_sources/topics-rnd5.xml \
  --qrels trec_covid_sources/qrels-covid_d5_j0.5-5.txt \
  --archive /path/to/trec-covid.zip --out trec_covid_transfer_selection
sha256sum -c TREC_COVID_INFERENCE_FROZEN_SHA256.txt
sha256sum -c TREC_COVID_RESULTS_SHA256.txt
python3 verify_trec_covid_selection.py --topics trec_covid_sources/topics-rnd5.xml \
  --qrels trec_covid_sources/qrels-covid_d5_j0.5-5.txt \
  --archive /path/to/trec-covid.zip --selection trec_covid_transfer_selection
```

The selector refuses an archive whose MD5 differs. The frozen hashes also
verify that regenerated input bytes match those used in the experiments.
Use a new extraction; never overwrite an existing user's sample or directory.
See `README.md` and `trec_covid_sources/SOURCES.md` for further verification,
runtime limitations and underlying article-rights details.

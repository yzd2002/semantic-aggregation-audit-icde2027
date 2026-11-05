# Response-provenance supplement

This companion notice explains that early product-study response strings were saved after whitespace trimming. It contains no model response or source document text. `NOTICE.md` gives the exact disclosure; `recorded_response_provenance_check.py` and `response_provenance_check.json` record the source and saved-string audit. The script requires the main `02_reproducibility/` files and the extracted ACORD reconstruction candidate. All source/data rights and privacy still require author review.

```bash
python recorded_response_provenance_check.py --repro /path/to/02_reproducibility --acord /path/to/acord_reconstructable --out /path/to/new_check.json
```

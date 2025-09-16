#!/usr/bin/env bash
# Example: download PDXE processed annotation hosted on figshare (example)
mkdir -p data/external
# Novartis PDXE sample annotations (figshare)
curl -L -o data/external/pdxe_sample_annotations.txt "https://figshare.com/ndownloader/files/13331072"
# NOTE: For PDMR/PDXNet, follow portal download instructions; often requires account and large files.
# See README for PDMR / PDXNet links.


"""
A script that cleans up my bibtex.
"""

import bibtexparser
from bibtexparser.bparser import BibTexParser
import fire


def main(
    ifile='ref.bib',  # Original file.
    ofile='ref-clean.bib',  # Fixed file.
):
    """Fix duplication issues."""
    with open(ifile, 'r') as bibtex_file:
        bib_database = bibtexparser.load(
            bibtex_file, parser=BibTexParser(interpolate_strings=False)
        )

    # Dedup based on ID and title.
    cleanlist = []
    cleanids = set()
    titles = set()
    for entry in bib_database.entries:
        if 'title' not in entry:
            continue
        ID = entry['ID']
        title = entry['title'].lower()
        if ID not in cleanids and title not in titles:
            cleanids.add(ID)
            cleanlist.append(entry)
            titles.add(title)

    out_database = bibtexparser.bibdatabase.BibDatabase
    out_database.entries = cleanlist
    with open(ofile, 'w') as bibtex_file:
        bibtexparser.dump(out_database, bibtex_file)


if __name__ == "__main__":
    fire.Fire(main)

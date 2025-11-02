import csv
import sys
from collections import Counter

PATH = 'data/sample_data.csv'

def main():
    try:
        with open(PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f'ERROR: Could not read {PATH}: {e}', file=sys.stderr)
        return 2

    if not rows:
        print('File is empty')
        return 1

    # Print first 8 rows for quick inspection
    print('First rows:')
    for i, r in enumerate(rows[:8]):
        print(i, r)

    # Check header
    header = rows[0]
    if len(header) != 2 or header[0].strip().lower() != 'text' or header[1].strip().lower() != 'lang':
        print('WARNING: Unexpected header row; expected "text,lang"')

    # Build counts, skip any comment lines starting with #
    counts = Counter()
    bad = []
    for i, r in enumerate(rows[1:], start=2):
        if not r:
            continue
        # allow comment lines
        if len(r) == 1 and r[0].strip().startswith('#'):
            continue
        if len(r) != 2:
            bad.append((i, r))
            continue
        text, lang = r
        if not text.strip():
            bad.append((i, r))
            continue
        counts[lang.strip()] += 1

    print('\nLanguage counts:')
    for lang, c in counts.most_common():
        print(f'  {lang}: {c}')

    if bad:
        print('\nMalformed rows:')
        for i, r in bad[:20]:
            print(f'  line {i}: {r}')
        return 3

    print('\nCSV looks OK')
    return 0

if __name__ == '__main__':
    sys.exit(main())

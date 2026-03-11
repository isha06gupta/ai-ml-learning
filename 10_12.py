import csv
import sys
csv.field_size_limit(sys.maxsize)
import csv
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")

INPUT_FILE = "somefile.txt"

rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
n = len(lines)

while i < n:

    content_block = []

    # Collect content until we see a line followed by 'unrelated'
    while i + 1 < n and lines[i+1].lower() != "unrelated":
        content_block.append(lines[i])
        i += 1

    content = " ".join(content_block).strip()

    # Now the next line is the HEADLINE
    if i < n:
        headline = lines[i].strip()
        i += 1
    else:
        break

    # Now next line is LABEL (always 'unrelated')
    if i < n and lines[i].lower() == "unrelated":
        label = "unrelated"
        i += 1
    else:
        break

    rows.append([content, headline, label])


# ==========================
# WRITE FORMATTED CSV
# ==========================
with open("formatted_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["NewsBodyContent", "NewsBodyHeadline", "Label"])
    writer.writerows(rows)

print("✓ Step 1 fixed: formatted_data.csv created CLEAN")


# ==========================
# WORD COUNTS
# ==========================
with open("formatted_data.csv", "r", encoding="utf-8") as infile, \
     open("word_count.csv", "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile, delimiter="\t")
    writer = csv.writer(outfile, delimiter="\t")

    header = next(reader)
    writer.writerow(header + ["Content_WordCount", "Headline_WordCount"])

    for content, headline, label in reader:
        writer.writerow([content, headline, label,
                         len(content.split()), len(headline.split())])

print("✓ Step 2: word_count.csv created")


# ==========================
# TOKEN COUNTS
# ==========================
with open("formatted_data.csv", "r", encoding="utf-8") as infile, \
     open("token_count.csv", "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(infile, delimiter="\t")
    writer = csv.writer(outfile, delimiter="\t")

    header = next(reader)
    writer.writerow(header + ["Content_TokenCount", "Headline_TokenCount"])

    for content, headline, label in reader:
        writer.writerow([content, headline, label,
                         len(word_tokenize(content)),
                         len(word_tokenize(headline))])

print("✓ Step 3: token_count.csv created")
print("ALL DONE.")


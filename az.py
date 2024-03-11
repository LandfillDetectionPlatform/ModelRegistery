# Define a function to clean each line from unwanted characters
def clean_line(line):
    # Strip the line of whitespace and split by '==' to get the package name
    # Also remove any unusual characters by encoding to a limited character set and decoding it back
    return line.strip().split('==')[0].encode('ascii', 'ignore').decode()

# Read the original file, clean each line, and collect cleaned package names
with open("requirements.txt", "r", encoding="utf-8") as readFile:
    packages_without_versions = [clean_line(pkg) for pkg in readFile.read().splitlines()]

# Write the cleaned package names back to the file
with open("rquirements_clean.txt", "w", encoding="utf-8") as writeFile:
    for pkg in packages_without_versions:
        writeFile.write(f"{pkg}\n")

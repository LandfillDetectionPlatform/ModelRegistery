# Open the requirements.txt file in read mode
with open("requirements.txt") as myFile:
    # Read the file's content
    pkgs = myFile.read()
    # Split the content into lines
    pkgs = pkgs.splitlines()

    # Iterate over each line
    for pkg in pkgs:
        # Split the package name from its version and print the package name
        print(pkg.split('==')[0])

import sys
from typing import NoReturn

def exit_with_error(message: str, details: str = "") -> NoReturn:
    # Message printing and exit code kept from the way it worked using the OptionParser (in case
    # someone parses the error output)
    sys.stderr.write("Usage: gdal2tiles [options] input_file [output]\n\n")
    sys.stderr.write("gdal2tiles: error: %s\n" % message)
    if details:
        sys.stderr.write("\n\n%s\n" % details)

    sys.exit(2)

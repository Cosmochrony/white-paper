#!/bin/bash

# White paper compilation script
# Compiles Cosmochrony.tex with bibliography support

set -e  # Exit on error

# Configuration
TEX_FILE="tex/Cosmochrony.tex"
OUTPUT_DIR="out"
MAIN_NAME="Cosmochrony"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Compiling white paper ===${NC}"

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Creating output directory: $OUTPUT_DIR${NC}"
    mkdir -p "$OUTPUT_DIR"
fi

# Set TEXINPUTS to include tex directory for .cls and .bib files
export TEXINPUTS=".:./tex:${TEXINPUTS}"

# Step 1: First pdflatex run
echo -e "${GREEN}Step 1/4: First pdflatex run${NC}"
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
    -output-directory="$OUTPUT_DIR" \
    -recorder \
    "$TEX_FILE"

# Step 2: Run bibtex
echo -e "${GREEN}Step 2/4: Running bibtex${NC}"
cd "$OUTPUT_DIR"
BSTINPUTS="../tex:${BSTINPUTS}" BIBINPUTS="../tex:${BIBINPUTS}" bibtex "$MAIN_NAME"
cd ..

# Step 3: Second pdflatex run (for bibliography)
echo -e "${GREEN}Step 3/4: Second pdflatex run${NC}"
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
    -output-directory="$OUTPUT_DIR" \
    -recorder \
    "$TEX_FILE"

# Step 4: Third pdflatex run (for cross-references)
echo -e "${GREEN}Step 4/4: Third pdflatex run${NC}"
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
    -output-directory="$OUTPUT_DIR" \
    -recorder \
    "$TEX_FILE"

# Final check
if [ -f "$OUTPUT_DIR/$MAIN_NAME.pdf" ]; then
    echo -e "${GREEN}=== Compilation successful ===${NC}"
    echo -e "Output: $OUTPUT_DIR/$MAIN_NAME.pdf"
    ls -lh "$OUTPUT_DIR/$MAIN_NAME.pdf"
else
    echo -e "${RED}=== Compilation failed ===${NC}"
    echo -e "Check $OUTPUT_DIR/$MAIN_NAME.log for errors"
    exit 1
fi

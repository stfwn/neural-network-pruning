# export.sh
#
# Call like this: sh export.sh <filename_of_slides>
# Then open slides.pdf in your PDF viewer.
#
# You need Pandoc and a LaTeX installation.

pandoc $1 -t beamer -o slides.pdf --bibliography=refs.bib

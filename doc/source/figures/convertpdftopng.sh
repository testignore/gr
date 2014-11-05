#!/bin/bash

if [ $# -ne 3 ]; then
	echo "usage: convertpdftopng.sh pdffigure.pdf pngfigure.png width"
	exit 1
fi

gs -dBATCH -dNOPAUSE -sDEVICE=pnmraw -r300x300 -sOutputFile=temp.pnm "$1"
convert temp.pnm +trim +repage -scale $3 $2
rm -f temp.pnm magick*


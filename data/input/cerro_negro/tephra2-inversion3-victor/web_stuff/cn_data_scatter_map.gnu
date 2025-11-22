set terminal pngcairo enhanced size 800,600
set output 'scatter_map_cn_data.png'

set title "Observed mass loading"
set xlabel "easting (m)"
set ylabel "northing (m)"
set format x "%.0f"
set format y "%.0f"
set grid
set size ratio -1  # Ensures equal axis scaling
set style fill solid
# Scale factor for circle size (adjust as needed)
scale_factor = 0.2  

# Use solid filled circles with black borders
set style fill solid border lc rgb "black"

plot 'cn_out.dat' using 1:2:(scale_factor * abs($4)) with circles fc rgb "gray" notitle, \
'-' using 1:2 with points pt 9 ps 2 lc rgb "black" notitle
532400 1382525
e


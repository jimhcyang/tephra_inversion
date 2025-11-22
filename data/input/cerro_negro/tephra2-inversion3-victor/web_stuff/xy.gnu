set terminal pngcairo size 800,600
set output 'colht_gainsize.png'

set title "Maximum Column Height vs Median Grainsize)"
set ylabel "Median Grainsize (phi)"
set xlabel "Maximum Column Height (m)"
set grid

# Plot using red circles with black outlines
plot 'cn_out.dat' using 4:3 with points pt 7 lc rgb "red" lw 1.5 lt -1 ps 1.5 notitle


set terminal pngcairo enhanced size 800,600
set output 'equiline.png'

set xlabel "{/Symbol \326} observed mass loading (kg/m^2)"
set ylabel "{/Symbol \326}  calculated mass loading (kg/m^2)"

set grid
set style data points  # Ensure scatter plot style

plot 'cn_out.dat' using (sqrt($4)):(sqrt($5)) with points pointtype 7 pointsize 1 notitle,\
     '-' using 1:2 with lines lw 2 lc rgb "red" notitle
0 0
45 45
e     


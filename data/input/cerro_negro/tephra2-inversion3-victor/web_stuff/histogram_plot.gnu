# Set the input data file
datafile = 'cn_out.dat'

# Set output to PNG file
set terminal pngcairo size 800, 600 enhanced
set output 'histogram_binning.png'

# Set style for histogram
set style fill solid 0.5  # Solid fill with 50% opacity
set boxwidth 5     # Bin width is 0.1

# Configure axes
set xlabel 'RMSE (kg/m^2)'
set ylabel 'Frequency'


# Set x-axis range explicitly
set xrange [200:260]

# Define bin width
bin_width = 5

# Define binning function
bin(x, width) = width * floor(x / width) + width / 2.0

# Plot histogram
plot datafile using (bin($5, bin_width)):(1.0) smooth frequency with boxes lc rgb "blue" notitle

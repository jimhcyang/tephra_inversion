
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
<meta name="description" content="Inverting tephra fallout deposit data">
<meta name="author" content="Charles Connor" >
<meta name="keywords" content="geology, geophysics, volcanology, C, inversion, volcanology, tephra2, inversion, eruption mass, eruption source parameter">
<title>Geocomputation</title>

<!-- Bootstrap CSS -->
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk" crossorigin="anonymous">

<link rel='stylesheet' href=https://gscommunitycodes.usf.edu/geoscicommunitycodes/css/gscc.css><script src=https://gscommunitycodes.usf.edu/geoscicommunitycodes/scripts/mathjax-config.js async></script></head>
<?php echo "<link rel='stylesheet' href=https://" . $_SERVER['HTTP_HOST'] . "/geoscicommunitycodes/css/gscc.css>"; ?>

    
</head>
<body id="top">
<div id="page-container">
<div id="content-wrap">
<div id="navbar"></div>

<div class="jumbotron" id="gcc" class="jumbotron-fluid">
<div class="container pt-3 pb-3" >
<h1 class="display-4">Inverting tephra fallout deposit data</h1>

<h6>Learn about modifying Tephra2 to estimate many eruption source parameters for the 1992 Cerro Negro (Nicaragua) eruption using simulated annealing</h6>

</div>
</div>

<div class="container pt-4 pb-2" >
<h4 class="text-secondary">Contents</h3>
<ul>

<li><a href="#anneal">The simulated annealing method with multiple parameters</a></li>
<li><a href="#code">The C code, output and analysis</a></li>
<li><a href="#try">Things to try</a></li>
<li><a href="#refs">Some references on inversion of tephra fallout deposits</a></li>
</ul>
</div>

<div class="container">
<h3 class="text-dark">Introduction</h3>
<p>
Now we modify the inversion code to include four parameters that are estimated simultaneously: total eruption mass, maximum eruption column height, "alpha", and median grainsize of the deposit. This analysis still uses simulated annealing, but is now a multi-parameter estimation problem. In addition, this analysis pays more attention to parameter uncertainty. This is achieved by running the inversion many times, say 100 times, and finding the range of parameters that result in solutions with reasonable small root-mean-square-error (RMSE). </p>
</div>


<div id = "anneal" class="container pt-2 pb-2" >
<a href="#top"><i>[Top]</i></a>

<h3 class="text-dark">The Simulated Annealing Search to Estimate Four Parameters</h3>
<p>The same overall code structure is maintained in this example. Primarily, the simulated annealing function is modified to include search for four parameters: total eruption mass, maximum eruption column height, "alpha", and median grainsize of the deposit. Any one inversion produces a set of these four parameters and an RMSE value. Because simulated annealing uses a random number generator to select neighbor solutions, the best-fit parameters will change each time the code is executed. So, by running the inversion many times, the uncertainty in the eruption source parameters can be estimated. </p>

<p>In this example, the inversion is executed 100 times in order to get a sense of parameter uncertainty. For each simulation, the "optimal" parameter set is printed to output. Since the SA algorithm uses a random seed to start the simulation, each time a different solution is identified and printed. The flow chart for this procedure is:</p>

<img src="./images/tephra2_sim3_flowchart2.png" alt="sim flowchart" style="width: 100%; max-width: 750px;" />

<p>The key steps are the same as before, just with an additional parameter to be estimated:
<ul>

<li>Initialize the Bounds. Note the bounds on all parameters are now set in a data structure in main ()) </li>
<li>Specify the initial and final temperature of the calculation, the cooling rate, and the maximum number of iterations allowed. These parameters control the search for the best-fit total eruption mass and alpha values.
</li>
<li>Iterative Optimization:
<ul>
<li>Step 1: Choose a neighboring total eruption mass parameter, adjacent to the "current total eruption mass". The neighbor eruption mass is:
$$m_n = m_c + \left [ \frac{\textrm{rand()}}{\textrm{RAND_MAX}} - 0.5\right ] \times (m_{max} - m_{min}) \times 0.1$$
The neighbor alpha, column height and median grainsize are found in exactly the same way:

<li>Step 2: Compute the new tephra fallout mass loading at each observation point with the neighbor parameter values and update the RSME
<li> Step 3: Neighbor parameter values become the current parameter values if either the new RSME is lower or the RMSE is worse based on the Metropolis criterion:
$$\exp(-\Delta \textrm{RSME}/T) > \frac{\textrm{rand()}}{\textrm{RAND_MAX}}  $$
which means that nearby <i>worse</i> depth estimates are accepted (temporarily) with some probability that depends on the temperature ($T$). This approach helps the SA algorithm escape from local minima in the RMSE function (for example because the data are noisy).
<li> If the current parameter values are updated, then the temperature ($T$) is reduced by a factor of $\alpha_T$:
$$T = \alpha_T T $$
Usually $\alpha_T = 0.9$ to gradually decrease the temperature. So, the value of $T$ changes throughout the search. Early on in the search, $T$ is large, so "bad" moves are accepted leading to exploration of the entire range of possible eruption mass and alpha values. As the search continues, $T$ becomes smaller and the search focuses only on better moves (lower RMSE values).
<li>Return the Optimal total eruption mass and optimal alpha when the $T_{final}$ value is reached.
</ul>
</p>

</div>

<div id = "code" class="container pt-2 pb-2" >
<a href="#top"><i>[Top]</i></a>
<h3 class="text-dark">The <i>C</i> code</h3>
<p>You need a bunch of files to compile and run this code. These files are contained in a compressed file called <a href = "./tephra2-inversion3-victor.zip">tephra2-inversion3-victor.zip</a>.
  
</p>

<p>Download this compressed file and unzip. To compile the code, on the command line type: make</p>

<p>To run the code type: ./tephra2_2025 tephra2.conf cerro_negro_92.dat wind1 > cn_out.dat</p>

<p> If all goes well, the first lines of your output file should look like:</p>
<div class="container bg-light">

<pre class="pre-scrollable">
<code class="text-success">
# eruption_mass, alpha, median phi, max plume height, rmse
30021943727.285576 3.180860 -0.056335 7155.299129 252.018181
41108341352.133232 3.395728 0.873463 7313.697391 218.149899
27261214988.893471 3.604860 -0.553366 8866.659696 240.702996
35450434898.701698 3.902670 0.603898 6110.914288 228.479505
29244515074.763687 3.168427 -0.032769 7527.717714 246.658578
34611805124.260399 3.723994 -0.074945 8240.863971 227.447151
39764786277.322464 4.091041 0.784999 7264.116956 216.192832
30071700615.608921 3.679748 0.037787 6703.240077 242.700108
</code>
</pre>
</div>
<p>Parameter sets are printed for each "optimal" solution along with the RMSE for that solution. Note that your solutions will not be exactly the same, since a random seed uis used in the code to start the SA search.</p>

<p> You can explore the output by plotting histograms of parameter values and by plotting scatter plots to search for correlation among estimated parameter values:</p>

<figure class="half" style="display:flex">
    <img style="width:400px" src="./images/mass_histo.png" alt="mass histogram">
    <img style="width:400px" src="./images/max_plume_ht_histo.png" alt="max plume histogram">
    <figcaption>Eruption mass histogram (left) and maximum plume height histogram (right). The simulations yield reasonable ranges for total eruption mass and maximum eruption column height. Both parameter estimates show some central tendency. </figcaption>
</figure>



<figure class="half" style="display:flex">
    <img style="width:400px" src="./images/median_grainsize.png" alt="median phi histogram">
    <img style="width:400px" src="./images/alpha.png" alt="alpha histogram">
    <figcaption>Median of the total eruption grainsize distribution (left) and alpha (right). The simulations yield reasonable ranges for these ESPs. The median grainsize has central tendency to about +0.2 phi. Alpha has a best estimate of around 3.6, meaning most of the mass is concentrated relatively near the top of the eruption column.</figcaption>
</figure>


<figure class="half" style="display:flex">
    <img style="width:400px" src="./images/mass_colht.png" alt="mass vs plume height">
    <img style="width:400px" src="./images/mass_gainsize.png" alt="mass vs. grainsize">
    <figcaption>It is also important to search for correlation among the estimated eruption source parameters. In this case, there is no correlation between eruption mass and maximum plume height. There is a clear correlation between eruption mass and median grainsize. As the eruption mass becomes larger, median grainsize tends toward finer grainsize (larger positive phi values). Since finer grainsizes are more widely dispersed compared to coarser grainsizes (due to their slower settling velocities), the eruption mass can increase. In other words, more mass is deposited in distal locations by these simulations. </figcaption>
</figure>


<figure class="half" style="display:flex">
    <img style="width:400px" src="./images/colht_gainsize.png" alt="plume height vs grainsize">
    <img style="width:400px" src="./images/alpha_gainsize.png" alt="alpha vs grainsize">
    <figcaption>Likewise, a slight negative correlation is observed between maximum eruption column height and median grainsize. As the eruption column height parameter increases, the best-fit median grainsize becomes coarser. This makes sense, as coarser particles fall at greater speed from higher in the column. There is no observed correlation between grainsize and alpha, the eruption column shape parameter. </figcaption>
</figure>


<hr>
</div>

<div id = "try" class="container pt-2 pb-2" >
<a href="#top"><i>[Top]</i></a>

<h3 class="text-dark">Things to Try</h3>

<ul>
<li>Make sure you can download, compile and run the code. Running this code take some time (15 minutes?). Plot up the output to find the parameter ranges, central tendency in parameter values and any correlation among parameters.</li>

<li>In the current version of the code, the parameters are specified in main(). Try modifying the code so these parameter ranges are read from a configuration file. Hint: see the C shortcourse for an example of reading a configuration file: <a href="https://gscommunitycodes.usf.edu/geoscicommunitycodes/public/c-shortcourse/c-first-examples/data4_c.php">data4.c</a></li>
</ul>
<hr>
</div>


<div id = "refs" class="container pt-2 pb-2" >
<a href="#top"><i>[Top]</i></a>
<h3 class="text-dark">References</h3>
<p>
<ul>
<li> Connor, L.J. and Connor, C.B., 2006. Inversion is the key to dispersion: understanding eruption dynamics by inverting tephra fallout. Statistics in Volcanology, Geological Society of London, Special Publications, https://doi.org/10.1144/IAVCEI001.18. <i>Inversion using the 1992 Cerro Negro eruption as an example. </i> <a href="https://pubs.geoscienceworld.org/gsl/books/edited-volume/1732/chapter-abstract/107601115/Inversion-is-the-key-to-dispersionunderstanding?redirectedFrom=fulltext">link to article</a>
<li>White, J.T., Connor, C.B., Connor, L. and Hasenaka, T., 2017. Efficient inversion and uncertainty quantification of a tephra fallout model. Journal of Geophysical Research: Solid Earth, 122(1), pp.281-294. <i>Inversion of tephra fallout data from Cerro Negro and Kirishima eruptions using uncertainty quantification</i><a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JB013682"> link to article</a>
<li>Mannen, K., Hasenaka, T., Higuchi, A., Kiyosugi, K. and Miyabuchi, Y., 2020. Simulations of tephra fall deposits from a bending eruption plume and the optimum model for particle release. Journal of Geophysical Research: Solid Earth, 125(6), p.e2019JB018902. <i>Tephra2 inversion compared with a modified forward solution for a bent-over plume. </i><a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JB018902"> link to article </a>.
<li>Yang, Q., Pitman, E.B., Bursik, M. and Jenkins, S.F., 2021. Tephra deposit inversion by coupling Tephra2 with the Metropolis-Hastings algorithm: algorithm introduction and demonstration with synthetic datasets. Journal of Applied Volcanology, 10, pp.1-24. <i>Inversion using a Metropolis-Hastings algorithm</i>. <a href="https://link.springer.com/article/10.1186/s13617-020-00101-4"> link to article </a>.
<li>Constantinescu, R., White, J.T., Connor, C.B., Hopulele‐Gligor, A., Charbonnier, S., Thouret, J.C., Lindsay, J.M. and Bertin, D., 2022. Uncertainty quantification of eruption source parameters estimated from tephra fall deposits. Geophysical Research Letters, 49(6), p.e2021GL097425. <i>Inversion modeling larger eruptions with a modified forward solution</i>. <a href = "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL097425"> link to article</a>
</ul>

 </p>
</div>

<div id="footer"></div>

<!-- Bootstrap core JavaScript
================================================== -->
<!-- Placed at the end of the document so the pages load faster -->
<!-- JS, Popper.js, and jQuery -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js" integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/p5@1.0.0/lib/p5.js"></script>
<script src="sphere/p5/canvas_gravity_sphere.js"></script>
<script src=https://gscommunitycodes.usf.edu/geoscicommunitycodes/scripts/gravity_scripts.js></script></body>
<?php echo "<script src=https://" . $_SERVER['HTTP_HOST'] . "/geoscicommunitycodes/scripts/shortcourse_scripts.js></script>"; ?>
</div>
</div>
</div>
</body>
</html>

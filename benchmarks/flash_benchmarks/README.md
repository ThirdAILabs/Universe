
# Instructions for running the image search benchmark:
The data is stored on the Blade server, so either run this on the blade server 
(in /media/scratch/ImageNetDemo/IndexFiles/) or copy the data files from there to a 
local folder (you can just copy the entire folder).

To get results, run python3 image_search.py > results.txt and then
run image_results_parse.py. This will parse the results file and then print a Pareto
plot of recall (R10@100 vs time). The R10@100 statistic is specified in image_search.py.

Note this describes an example of how to use MagSearch and is not a production demo.

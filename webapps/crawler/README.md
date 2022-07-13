# Web Crawler Demo

For running the crawler 
- Make sure you have scrapy installed. Use pip3 install scrapy for installing
- cd to the folder "webappps/crawler"
- In the python script crawler_args.py, enter the crawl parameters. This will create a json file named "arguments_for_crawler.json". 
- run python3 crawler_args.py 
- run the command "scrapy crawl simple_crawl -a args_file="arguments_for_crawler.json" -O dynamic_parsing.jl"

the crawler will crawl the websites and write crawl metadata such as URLs parsed to "dynamic_parsing.jl" 
you can also use the .jl for debugging purposes by setting "parse dynamic" as true in crawler_args.py
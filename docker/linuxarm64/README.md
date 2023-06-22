### Building a Wheel for x86_64
1. `docker build --memory=4g -t universe-x86 -f docker/linuxarm64/Dockerfile --platform linux/x86_64 .`
2. `docker run -v /PATH/TO/VOLUME/:/volume universe-x86`

### Building a Wheel for arm64
1. `docker build --memory=4g -t universe-arm64 -f docker/linuxarm64/Dockerfile  .`
2. `docker run -v /PATH/TO/VOLUME/:/volume universe-arm64`
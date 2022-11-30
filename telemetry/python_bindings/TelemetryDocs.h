#pragma once

namespace thirdai::telemetry::python::docs {

const char* const START_METRICS = R"pbdoc(
Start a Prometheus metrics client on the passed in port. If a port is not 
specified this method will use the default ThirdAI port of 9929. This function
is not thread safe with other ThirdAI code, so you should make sure that no
other code is running when this method is called.
)pbdoc";

const char* const STOP_METRICS = R"pbdoc(
Stops the current Prometheus metrics client if one is running. This function
is not thread safe with other ThirdAI code, so you should make sure that no
other code is running when this method is called.
)pbdoc";

}  // namespace thirdai::telemetry::python::docs
# JFM v2 Golden Package

These hex files freeze a two-layer Qwen-shaped JFM v2 package independently
of the current test writer. `engine_test.cpp` decodes them and opens both
stage ranges. Update them only through an explicit format-version change.

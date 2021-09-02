CXX	:= g++

SRC_DIR   :=  ./src
BUILD_DIR := 	./bin
PY_BUILD_DIR := ./py_bin

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

PY_OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(PY_BUILD_DIR)/%.o)

TARGET := Bolt.cpp
BINARY := $(TARGET:.cpp=)

# INC_FLAGS := -I/usr/local/include
# LIB_FLAGS := -L/usr/local/lib

# -Ofast all optimizations of O3 plus bonus items
# -DNDEBUG disables assertions
# -fopenmp enables openmp library
# -march=native generates code for the cpu compiling the program and preforms optimizations on that ISA
# -fPIC generates position independent code
# -ffast-math faster but less precise math
# -funroll-loops unrolls loops with a fixed number of iterations at compile time
# -ftree-vectorize enables vectorization
CXX_OPT_FLAGS := -std=c++14 -Ofast -DNDEBUG -fopenmp -march=native -fPIC -ffast-math -funroll-loops -ftree-vectorize 

# Add -g flag for debugging symbols 
CXX_DBG_FLAGS := -Wall -Wextra -Werror

CXX_FLAGS := $(INC_FLAGS) $(LIB_FLAGS) $(CXX_OPT_FLAGS) $(CXX_DBG_FLAGS)

$(BINARY) : $(BUILD_DIR) $(OBJS)
	$(CXX) $(CXX_FLAGS) $(TARGET) $(OBJS) -o $@ 

$(OBJS) : $(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR): 
	@mkdir -p $(BUILD_DIR)

PYBIND_SRC := python_bindings/Bolt.cpp
PYBIND_TARGET := thirdai.so

ifeq ($(shell uname), Darwin) 
	ifeq ($(shell uname -m), arm64)
		PYTHON_FLAGS := -L/opt/homebrew/Cellar/python@3.9/3.9.6/Frameworks/Python.framework/Versions/3.9/lib -lpython3.9
	else 
		PYTHON_FLAGS := -L/usr/local/Cellar/python@3.9/3.9.6/Frameworks/Python.framework/Versions/3.9/lib -lpython3.9
	endif
endif

PYBIND_FLAGS := $(shell python3 -m pybind11 --includes)

bindings: $(PY_BUILD_DIR) $(PY_OBJS)
	$(CXX) $(CXX_FLAGS) $(PYBIND_FLAGS) $(PYTHON_FLAGS) $(PY_OBJS) --shared -fPIC $(PYBIND_SRC) -o $(PYBIND_TARGET)

.PHONY: bindings

$(PY_OBJS) : $(PY_BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXX_FLAGS) $(PYTHON_FLAGS) $(PYBIND_FLAGS) -DPYTHON_BINDINGS -c $< -o $@

$(PY_BUILD_DIR):
	@mkdir -p $(PY_BUILD_DIR)

clean: 
	rm -rf $(BUILD_DIR) $(BINARY) $(PY_BUILD_DIR) $(PYBIND_TARGET) shuffle

.PHONY: clean

shuffle:
	$(CXX) $(CXX_FLAGS) shuffle.cpp -o shuffle

.PHONY: shuffle

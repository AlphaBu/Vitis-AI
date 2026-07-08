##########################################################################
 # Copyright (C) 2025 Advanced Micro Devices, Inc.
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
###########################################################################

.DEFAULT_GOAL:=all

check_dependencies:
	@pkg-config --atleast-version=2.6 xrt || (echo "Error: xrt version 2.6 or later is required."; exit 1)
	@pkg-config --atleast-version=0.1 vart-x || (echo "Error: vart-x version 0.1 or later is required."; exit 1)

CFLAGS+= -Wall -Wextra -Wpedantic -Wunused -Wformat -Wdeprecated -Wdisabled-optimization -Wuninitialized -std=c++17
CFLAGS+= `pkg-config --cflags vart-x`
CFLAGS+= `pkg-config --cflags xrt`
CFLAGS+= -I.

LDFLAGS+= `pkg-config --libs vart-x`
LDFLAGS+= `pkg-config --libs xrt`

SRCS:= pl_test.cpp

OBJDIR = build

# Target executable
TARGET = pl_test

OBJS = $(addprefix $(OBJDIR)/, $(notdir $(SRCS:.cpp=.o)))

# Linking the executable from object files
$(TARGET): $(OBJS)
	$(CXX) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

all: check_dependencies $(TARGET)

install:
	@echo "Installing $(TARGET) to $(SDKTARGETSYSROOT)"
	cp $(TARGET) $(SDKTARGETSYSROOT)/usr/bin/

# Ensure build directory exists
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Pattern rule for sources in current directory
$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CFLAGS) -c $< -o $@

# Clean target to remove generated files
clean:
	rm -rf $(OBJDIR) $(TARGET)

distclean:
	rm -f $(SDKTARGETSYSROOT)/usr/bin/$(TARGET)

# Phony targets
.PHONY: all check_dependencies clean install distclean

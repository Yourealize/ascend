# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ma-user/work/GroupNormV2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ma-user/work/GroupNormV2/build_out

# Utility rule file for ascendc_bin_ascend910b_group_norm_v2_0.

# Include the progress variables for this target.
include op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/progress.make

op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0:
	cd /home/ma-user/work/GroupNormV2/build_out/op_kernel/binary/ascend910b && export HI_PYTHON=python3 && bash /home/ma-user/work/GroupNormV2/build_out/op_kernel/binary/ascend910b/gen/GroupNormV2-group_norm_v2-0.sh /home/ma-user/work/GroupNormV2/build_out/op_kernel/binary/ascend910b/src/GroupNormV2.py /home/ma-user/work/GroupNormV2/build_out/op_kernel/binary/ascend910b/bin/group_norm_v2 && echo $(MAKE)

ascendc_bin_ascend910b_group_norm_v2_0: op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0
ascendc_bin_ascend910b_group_norm_v2_0: op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/build.make

.PHONY : ascendc_bin_ascend910b_group_norm_v2_0

# Rule to build all files generated by this target.
op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/build: ascendc_bin_ascend910b_group_norm_v2_0

.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/build

op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/clean:
	cd /home/ma-user/work/GroupNormV2/build_out/op_kernel && $(CMAKE_COMMAND) -P CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/cmake_clean.cmake
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/clean

op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/depend:
	cd /home/ma-user/work/GroupNormV2/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/work/GroupNormV2 /home/ma-user/work/GroupNormV2/op_kernel /home/ma-user/work/GroupNormV2/build_out /home/ma-user/work/GroupNormV2/build_out/op_kernel /home/ma-user/work/GroupNormV2/build_out/op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_kernel/CMakeFiles/ascendc_bin_ascend910b_group_norm_v2_0.dir/depend


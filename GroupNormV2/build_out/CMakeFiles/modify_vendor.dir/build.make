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

# Utility rule file for modify_vendor.

# Include the progress variables for this target.
include CMakeFiles/modify_vendor.dir/progress.make

CMakeFiles/modify_vendor: scripts/install.sh
CMakeFiles/modify_vendor: scripts/upgrade.sh


scripts/install.sh:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ma-user/work/GroupNormV2/build_out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating scripts/install.sh, scripts/upgrade.sh"
	mkdir -p /home/ma-user/work/GroupNormV2/build_out/scripts
	cp -r /home/ma-user/work/GroupNormV2/scripts/* /home/ma-user/work/GroupNormV2/build_out/scripts/
	sed -i s/vendor_name=customize/vendor_name=customize/g /home/ma-user/work/GroupNormV2/build_out/scripts/*

scripts/upgrade.sh: scripts/install.sh
	@$(CMAKE_COMMAND) -E touch_nocreate scripts/upgrade.sh

modify_vendor: CMakeFiles/modify_vendor
modify_vendor: scripts/install.sh
modify_vendor: scripts/upgrade.sh
modify_vendor: CMakeFiles/modify_vendor.dir/build.make

.PHONY : modify_vendor

# Rule to build all files generated by this target.
CMakeFiles/modify_vendor.dir/build: modify_vendor

.PHONY : CMakeFiles/modify_vendor.dir/build

CMakeFiles/modify_vendor.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/modify_vendor.dir/cmake_clean.cmake
.PHONY : CMakeFiles/modify_vendor.dir/clean

CMakeFiles/modify_vendor.dir/depend:
	cd /home/ma-user/work/GroupNormV2/build_out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/work/GroupNormV2 /home/ma-user/work/GroupNormV2 /home/ma-user/work/GroupNormV2/build_out /home/ma-user/work/GroupNormV2/build_out /home/ma-user/work/GroupNormV2/build_out/CMakeFiles/modify_vendor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/modify_vendor.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

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
CMAKE_SOURCE_DIR = /home/himlen/rgbd_seg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/himlen/rgbd_seg

# Include any dependencies generated for this target.
include CMakeFiles/rawpixel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rawpixel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rawpixel.dir/flags.make

CMakeFiles/rawpixel.dir/rawpixel.cpp.o: CMakeFiles/rawpixel.dir/flags.make
CMakeFiles/rawpixel.dir/rawpixel.cpp.o: rawpixel.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/himlen/rgbd_seg/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/rawpixel.dir/rawpixel.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/rawpixel.dir/rawpixel.cpp.o -c /home/himlen/rgbd_seg/rawpixel.cpp

CMakeFiles/rawpixel.dir/rawpixel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rawpixel.dir/rawpixel.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/himlen/rgbd_seg/rawpixel.cpp > CMakeFiles/rawpixel.dir/rawpixel.cpp.i

CMakeFiles/rawpixel.dir/rawpixel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rawpixel.dir/rawpixel.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/himlen/rgbd_seg/rawpixel.cpp -o CMakeFiles/rawpixel.dir/rawpixel.cpp.s

CMakeFiles/rawpixel.dir/rawpixel.cpp.o.requires:
.PHONY : CMakeFiles/rawpixel.dir/rawpixel.cpp.o.requires

CMakeFiles/rawpixel.dir/rawpixel.cpp.o.provides: CMakeFiles/rawpixel.dir/rawpixel.cpp.o.requires
	$(MAKE) -f CMakeFiles/rawpixel.dir/build.make CMakeFiles/rawpixel.dir/rawpixel.cpp.o.provides.build
.PHONY : CMakeFiles/rawpixel.dir/rawpixel.cpp.o.provides

CMakeFiles/rawpixel.dir/rawpixel.cpp.o.provides.build: CMakeFiles/rawpixel.dir/rawpixel.cpp.o

CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o: CMakeFiles/rawpixel.dir/flags.make
CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o: gco/GCoptimization.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/himlen/rgbd_seg/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o -c /home/himlen/rgbd_seg/gco/GCoptimization.cpp

CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/himlen/rgbd_seg/gco/GCoptimization.cpp > CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.i

CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/himlen/rgbd_seg/gco/GCoptimization.cpp -o CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.s

CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o.requires:
.PHONY : CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o.requires

CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o.provides: CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o.requires
	$(MAKE) -f CMakeFiles/rawpixel.dir/build.make CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o.provides.build
.PHONY : CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o.provides

CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o.provides.build: CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o

CMakeFiles/rawpixel.dir/gco/graph.cpp.o: CMakeFiles/rawpixel.dir/flags.make
CMakeFiles/rawpixel.dir/gco/graph.cpp.o: gco/graph.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/himlen/rgbd_seg/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/rawpixel.dir/gco/graph.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/rawpixel.dir/gco/graph.cpp.o -c /home/himlen/rgbd_seg/gco/graph.cpp

CMakeFiles/rawpixel.dir/gco/graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rawpixel.dir/gco/graph.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/himlen/rgbd_seg/gco/graph.cpp > CMakeFiles/rawpixel.dir/gco/graph.cpp.i

CMakeFiles/rawpixel.dir/gco/graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rawpixel.dir/gco/graph.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/himlen/rgbd_seg/gco/graph.cpp -o CMakeFiles/rawpixel.dir/gco/graph.cpp.s

CMakeFiles/rawpixel.dir/gco/graph.cpp.o.requires:
.PHONY : CMakeFiles/rawpixel.dir/gco/graph.cpp.o.requires

CMakeFiles/rawpixel.dir/gco/graph.cpp.o.provides: CMakeFiles/rawpixel.dir/gco/graph.cpp.o.requires
	$(MAKE) -f CMakeFiles/rawpixel.dir/build.make CMakeFiles/rawpixel.dir/gco/graph.cpp.o.provides.build
.PHONY : CMakeFiles/rawpixel.dir/gco/graph.cpp.o.provides

CMakeFiles/rawpixel.dir/gco/graph.cpp.o.provides.build: CMakeFiles/rawpixel.dir/gco/graph.cpp.o

CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o: CMakeFiles/rawpixel.dir/flags.make
CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o: gco/LinkedBlockList.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/himlen/rgbd_seg/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o -c /home/himlen/rgbd_seg/gco/LinkedBlockList.cpp

CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/himlen/rgbd_seg/gco/LinkedBlockList.cpp > CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.i

CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/himlen/rgbd_seg/gco/LinkedBlockList.cpp -o CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.s

CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o.requires:
.PHONY : CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o.requires

CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o.provides: CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o.requires
	$(MAKE) -f CMakeFiles/rawpixel.dir/build.make CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o.provides.build
.PHONY : CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o.provides

CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o.provides.build: CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o

# Object files for target rawpixel
rawpixel_OBJECTS = \
"CMakeFiles/rawpixel.dir/rawpixel.cpp.o" \
"CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o" \
"CMakeFiles/rawpixel.dir/gco/graph.cpp.o" \
"CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o"

# External object files for target rawpixel
rawpixel_EXTERNAL_OBJECTS =

rawpixel: CMakeFiles/rawpixel.dir/rawpixel.cpp.o
rawpixel: CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o
rawpixel: CMakeFiles/rawpixel.dir/gco/graph.cpp.o
rawpixel: CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o
rawpixel: CMakeFiles/rawpixel.dir/build.make
rawpixel: /home/himlen/opencv3/build/lib/libopencv_stitching.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_superres.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_videostab.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_aruco.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_bgsegm.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_bioinspired.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_ccalib.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_dpm.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_freetype.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_fuzzy.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_hdf.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_line_descriptor.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_optflow.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_reg.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_saliency.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_sfm.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_stereo.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_structured_light.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_surface_matching.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_tracking.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_xfeatures2d.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_ximgproc.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_xobjdetect.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_xphoto.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_shape.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_viz.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_phase_unwrapping.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_rgbd.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_calib3d.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_video.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_datasets.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_dnn.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_face.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_plot.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_text.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_features2d.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_flann.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_objdetect.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_ml.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_highgui.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_photo.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_videoio.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_imgcodecs.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_imgproc.so.3.2.0
rawpixel: /home/himlen/opencv3/build/lib/libopencv_core.so.3.2.0
rawpixel: CMakeFiles/rawpixel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable rawpixel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rawpixel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rawpixel.dir/build: rawpixel
.PHONY : CMakeFiles/rawpixel.dir/build

CMakeFiles/rawpixel.dir/requires: CMakeFiles/rawpixel.dir/rawpixel.cpp.o.requires
CMakeFiles/rawpixel.dir/requires: CMakeFiles/rawpixel.dir/gco/GCoptimization.cpp.o.requires
CMakeFiles/rawpixel.dir/requires: CMakeFiles/rawpixel.dir/gco/graph.cpp.o.requires
CMakeFiles/rawpixel.dir/requires: CMakeFiles/rawpixel.dir/gco/LinkedBlockList.cpp.o.requires
.PHONY : CMakeFiles/rawpixel.dir/requires

CMakeFiles/rawpixel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rawpixel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rawpixel.dir/clean

CMakeFiles/rawpixel.dir/depend:
	cd /home/himlen/rgbd_seg && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/himlen/rgbd_seg /home/himlen/rgbd_seg /home/himlen/rgbd_seg /home/himlen/rgbd_seg /home/himlen/rgbd_seg/CMakeFiles/rawpixel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rawpixel.dir/depend

###############################################################################
 #  Copyright (c) 2016, Xilinx, Inc.
 #  All rights reserved.
 #
 #  Redistribution and use in source and binary forms, with or without
 #  modification, are permitted provided that the following conditions are met:
 #
 #  1.  Redistributions of source code must retain the above copyright notice,
 #     this list of conditions and the following disclaimer.
 #
 #  2.  Redistributions in binary form must reproduce the above copyright
 #      notice, this list of conditions and the following disclaimer in the
 #      documentation and/or other materials provided with the distribution.
 #
 #  3.  Neither the name of the copyright holder nor the names of its
 #      contributors may be used to endorse or promote products derived from
 #      this software without specific prior written permission.
 #
 #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 #  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 #  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 #  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 #  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 #  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 #  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 #  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 #  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 #  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
###############################################################################
###############################################################################
 #
 #
 # @file hls-syn.tcl
 #
 # Tcl script for HLS synthesis of the target network (this script is 
 # automatically launched when executing make-hw.sh script)
 #
 #
###############################################################################
# ignore the first 2 args, since Vivado HLS also passes -f tclname as args

set config_proj_name [lindex $argv 2]
puts "HLS project: $config_proj_name"
set config_hwsrcdir [lindex $argv 3]
puts "HW source dir: $config_hwsrcdir"
set directory_params [lindex $argv 4] 
set test_image [lindex $argv 5] 
set expected_result [lindex $argv 6] 
set config_proj_part [lindex $argv 7] 


set config_bnnlibdir "$::env(XILINX_BNN_ROOT)/library/hls"
set config_bnnhostlibdir "$::env(XILINX_BNN_ROOT)/library/host"
set config_tinycnn "$::env(XILINX_BNN_ROOT)/xilinx-tiny-cnn"
puts "BNN HLS library: $config_bnnlibdir"

set config_toplevelfxn "BlackBoxJam"
set config_clkperiod [lindex $argv 8] 

# set up project
open_project $config_proj_name
add_files $config_hwsrcdir/top.cpp -cflags "-std=c++0x -I$config_bnnlibdir"

add_files -tb $config_hwsrcdir/../sw/main_python.cpp -cflags "-DOFFLOAD -DRAWHLS -std=c++0x -I$config_bnnhostlibdir -I$config_bnnlibdir -I$config_tinycnn -I$config_hwsrcdir"
add_files -tb $config_bnnhostlibdir/foldedmv-offload.cpp -cflags "-DOFFLOAD -DRAWHLS -std=c++0x -I$config_bnnhostlibdir -I$config_bnnlibdir -I$config_tinycnn"
add_files -tb $config_bnnhostlibdir/rawhls-offload.cpp -cflags "-DOFFLOAD -DRAWHLS -std=c++0x -I$config_bnnhostlibdir -I$config_bnnlibdir -I$config_tinycnn"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

# use 64-bit AXI MM addresses
config_interface -m_axi_addr64

# syntesize and export
create_clock -period $config_clkperiod -name default
csim_design -argv "$directory_params $test_image 10 $expected_result" -compiler clang
csynth_design
export_design -format ip_catalog
exit 0

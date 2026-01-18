// In Vivado 2022 don't include this line. On EdaPlaygrounds do.
import uvm_pkg::*; 

`include "uvm_macros.svh"
`include "test.svh"
`include "interface.svh"

// Top module in the hierarchy. includes all the testing
module top;

  // Instantiate the interface
  salu_dut dut_ifl();
  
  // Instantiate the DUT and connect it to the interface
  salu dut1 (
  // *** Asignment: connect the dut to the interface
  );
    
  initial begin
    // Place the interface into the UVM configuration database
    uvm_config_db#(virtual salu_dut)::set(null, "*", "dut_vif", dut_ifl);
    // Start the test
    run_test("salu_test");
  end
  
  // Dump waves
  initial begin
    $dumpfile("dump.vcd");
    $dumpvars(0, top);
  end
  
endmodule

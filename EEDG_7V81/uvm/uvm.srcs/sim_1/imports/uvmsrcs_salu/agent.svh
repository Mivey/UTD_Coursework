// The uvm sequence, transaction item, and driver are in these files:
`include "sequencer.svh"
`include "monitor.svh"
`include "driver.svh"

// The agent contains sequencer, driver, and monitor
class salu_agent extends uvm_agent;
   
   // *** Asignment: register agent using uvm_macros
  `uvm_component_utils(salu_agent)
  // *** Asignment: declare the monitor and driver objects
  salu_dut driver;
  salu_monitor monitor;
  // our sequencer initialized
  uvm_sequencer#(salu_packet) sequencer;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    // driver, sequencer, and monitor objects initialized
  super.build_phase(phase);
	// *** Asignment: initialize driver, sequencer, and monitor objects

  sequencer = uvm_sequencer#(salu_packet)::type_id::create("sequencer", this);
  driver = salu_dut::type_id::create("driver", this);
  monitor = salu_monitor::type_id::create("monitor", this);
  endfunction    

  // connect_phase of the agent
  function void connect_phase(uvm_phase phase);
	// *** Asignment: connect the driver and the sequencer
    super.connect_phase(phase);
    driver.seq_item_port.connnect(sequencer.seq_item_export);
	// make the monitor and driver drvdone events point to the same thing
    monitor.drvdone = driver.drvdone;
  endfunction

  task run_phase(uvm_phase phase);
    // We raise objection to keep the test from completing
    phase.raise_objection(this);


	// *** Assignment: create sequence and start it.


    // We drop objection to allow the test to complete
    phase.drop_objection(this);
  endtask

endclass

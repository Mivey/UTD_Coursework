
`include "agent.svh"
`include "scoreboard.svh"

// our environment class
class salu_env extends uvm_env;
  `uvm_component_utils(salu_env)

  // *** Asignment: instantiate the environment's constituents (scoreboard and monitor)
  salu_agent agent;
  salu_scoreboard scrbd;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
   // *** Asignment: initialize the environment's constituents
    super.build_phase(phase);
    agent = salu_agent::type_id::create("agent", this);
    scrbd = salu_scoreboard::type_id::create("scrbd", this);

  endfunction
  
  function void connect_phase(uvm_phase phase);
    // *** Asignment: connect the agent and the monitor's analysis_port to each other using the .connect function.
    super.connect_phase(phase);
    agent.monitor.mon_analysis_port.connect(scrbd.m_analysis_imp);
  endfunction

endclass
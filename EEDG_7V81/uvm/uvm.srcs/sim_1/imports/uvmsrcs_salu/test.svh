

`include "env.svh"

class salu_test extends uvm_test;
  
  // *** Asignment: use the correct uvm utils macro to register this object
`uvm_component_utils (salu_test)

  // *** Asignment: add an instance of the environment object to your test (recall that uvm_test encapsulates uvm_env)
  salu_env m_top_env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    // *** Asignment: initialize the environment object using type_id::create("env", this) syntax
    super.build_phase (phase);
    m_top_env = salu_env::type_id::create("m_top_env", this);
  endfunction

  task run_phase(uvm_phase phase);
     // the test run_phase is empty. The env and agent run_phase take care of things.
  endtask

endclass
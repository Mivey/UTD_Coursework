
// driver class. gets items from sequencer and drives them into the dut
class salu_dut extends uvm_driver #(salu_packet);

  `uvm_component_utils(salu_dut)
  
  virtual salu_dut dut_vif;
  event drvdone, mondone;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    if(!uvm_config_db#(virtual salu_dut)::get(this, "", "dut_vif", dut_vif)) begin
      `uvm_error("", "uvm_config_db::get failed")
    end
  endfunction 

  task run_phase(uvm_phase phase);
    
    // drive the packet into the dut
    forever begin
	  // *** Asignment: call get_next_item() on seq_item_port to get a packet. 2) configure the intreface signals using the packet.
      salu_packet pkt;

      
	  // packet pkt received from the seq_item_port is printed here
	  `uvm_info ("write", $sformatf("driving item a=%0b, b=%0b, cmd=%0b", pkt.a, pkt.b, pkt.cmd), UVM_MEDIUM)
	  seq_item_port.get_next_item(pkt);
	  // *** Asignment: configure the intreface signals using the packet. This involves place
	  // the values of a and b onto the interface and then toggling the clock.
    @(negedge dut_vif.clk);
    dut_vif.a <= pkt.a;
    dut_vif.a <= pkt.cmd;
    dut_vif.a <= pkt.b;
    @(negedge dut_vif.clk);
      
	  
	  // *** Asignment: call item_done() on the seq_item_port to let the sequencer know you are ready for the next item.
    seq_item_port.item_done();
      
	  
	  // *** Asignment: communicate to the monitor to let it know the signals have been written and the output is ready to received (use drvdone event)
    -> drvdone;
	  // *** Asignment: wait to make sure that monitor is done reading the output before loading the next item into the dut (you may create a new event, or simply wait some time)
      @(mondone);
    end
  endtask

endclass: salu_dut

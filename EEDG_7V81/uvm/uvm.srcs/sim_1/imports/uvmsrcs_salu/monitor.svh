
// our monitor class
class salu_monitor extends uvm_monitor;
   `uvm_component_utils (salu_monitor)
   
   virtual dut_if   vif;

   // this event is used to signal from the driver that a drive operation has concluded
   event drvdone, mondone;

   // this is the analysis port that is used to send the data to the scoreboard
   uvm_analysis_port #(salu_packet)   mon_analysis_port;
   
   function new (string name, uvm_component parent= null);
      super.new (name, parent);
   endfunction

   virtual function void build_phase (uvm_phase phase);
      super.build_phase (phase);

      // *** Assignment: Get virtual interface handle from the configuration DB
      if (!uvm_config_db#(virtual dut_if)::get(this,"", "dut_if", vif))
         `uvm_fatal("MON", "Didn't get the interface");
      // Create an instance of the analysis port
      mon_analysis_port = new ("mon_analysis_port", this);      
            
   endfunction

 virtual task run_phase (uvm_phase phase);
      salu_packet  data_obj = salu_packet::type_id::create ("data_obj", this);
      forever begin
	  
	  
	  // *** Asignment: you need to 1) wait for the drvdone event, 2) once the driver is done, reconstruct the data_obj packet by reading 
	  // the a,b,cmd, and most importantly the out fields of the interface
         @(drvdone);
         data_obj.a <= vif.a;
         data_obj.a <= vif.b;
         data_obj.a <= vif.cmd;
         data_obj.a <= vif.out;
         -> mondone;
      
	  
	  
	  // after reading the object from the interface print we read its contents
        `uvm_info ("write", $sformatf("driver done reading item a=%0b, b=%0b, cmd=%0b, o=%0b", data_obj.a, data_obj.b, data_obj.cmd, data_obj.out), UVM_MEDIUM)
		
		
       // *** Assignment: write data object to the analysis port to the scoreboard
         mon_analysis_port.write(data_obj);
       
      end
   endtask

endclass
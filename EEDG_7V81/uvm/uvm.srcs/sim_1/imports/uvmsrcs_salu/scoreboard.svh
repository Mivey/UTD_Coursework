
// scoreboard object
class salu_scoreboard extends uvm_scoreboard;
	`uvm_component_utils (salu_scoreboard)
	
	function new (string name = "salu_scoreboard", uvm_component parent);
		super.new (name, parent);
	endfunction

	// *** Asignment: define analysis port
	
	function void build_phase (uvm_phase phase);
		ap_imp = new ("ap_imp", this);
	endfunction
	
	// this function gets called when the monitor sends data to the scoreboard. We read the data and perform checks here
	virtual function void write (salu_packet data);
		// *** Asignment: write some checks to check whether the data.out result is equal 
		// to the ADD/AND/XOR/XNOR of data.a, and data.b when data.cmd is set to the right command
		`uvm_info ("write", $sformatf("Data received = 0x%0h", data.out), UVM_MEDIUM)
	endfunction

endclass

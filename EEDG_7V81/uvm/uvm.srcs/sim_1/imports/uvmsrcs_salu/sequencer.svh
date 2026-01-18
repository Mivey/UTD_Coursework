
class salu_packet extends uvm_sequence_item;

  `uvm_object_utils(salu_packet)

  parameter cmd_width = 3;
  parameter data_width = 16;

  logic[cmd_width-1:0] cmd;
  rand logic[data_width-1:0] a;
  rand logic[data_width-1:0] b;
  logic[data_width-1:0] out;

  constraint c1 { a < 20; a > 10;}
  constraint c2 {b < 20; b > 10; a > b;}


  // *** Asignment: write some constraints to keep the a and b in a small range (say 0 to 20)
  // *** Asignment: if your simulator does not support randomize() (such as ModelSim/QuestaSim) use the 
  // $urandom_range functions in the sequence body (seen bellow) to randomize your data in the range 0 to 20.
//  `uvm_object_utils_begin(salu_packet)
//    `uvm_feild_int (cmd, UVM_DEFAULT)
//    `uvm_feild_int (a, UVM_DEFAULT)
//    `uvm_feild_int (b, UVM_DEFAULT)
//    `uvm_feild_int (out, UVM_DEFAULT)
//  `uvm_object_utils_end

  function new (string name = "salu_packet");
    super.new(name);
  endfunction

  task automatic gen_op(int i);
    cmd = i;
  endtask //automatic

  task automatic gen_results();
    unique case (cmd)
      3'b000 : begin
        out = a + b;
      end
      3'b001 : begin
        out = a - b;
      end
      3'b010 : begin
        out = a ^ b;
      end
      3'b011 : begin
        out = a & b;
      end
      3'b100 : begin
        out = a | b;
      end
    endcase
    
  endtask //automatic

endclass: salu_packet

class salu_sequence extends uvm_sequence#(salu_packet);

  `uvm_object_utils(salu_sequence)

  function new (string name = "salu_sequence");
    super.new(name);
  endfunction

  task body;
	// body of the sequence. Where you must generate 5 random patterns for each of the 5 commands
	for (int i = 0; i < 5; i++) begin
	    `uvm_info ("write", $sformatf("sequence for cmd=%0b", i), UVM_MEDIUM);
		repeat(5) begin

      salu_packet m_salu = salu_packet::type_id::create(m_salu);
      start_item(m_salu);
      m_salu.randomize();
      m_salu.gen_op(i);
      m_salu.gen_results();
      finish_item(m_salu);

		  // *** Asignment: you need to 1) create a transaction packet object, 2) randomize its fields, 
		  // you will need to call start_item() and finish_item() on your transaction object 
		  // before and after randomization/configuration respectively;
		  // you should use the i variable to configure the cmd field. Remember only a and b are randomized.

		end
	end
  endtask: body

endclass: salu_sequence

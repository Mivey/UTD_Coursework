
module four_bits_tb;

  // Parameters

  //Ports
  reg [3:0] i_a;
  reg [3:0] i_b;
  reg  c_in;
  wire  c_out;
  wire [3:0] o_sum;

  four_bits  four_bits_inst (
    .i_a(i_a),
    .i_b(i_b),
    .c_in(c_in),
    .c_out(c_out),
    .o_sum(o_sum)
  );
  
  bit [8:0] jj;

//always #5  clk = ! clk ;
  initial begin
    i_a = 0;
    i_b = 0;
    c_in = 0;
    #1;
    $display("| Count\t\t|   a\t|   b\t|   c\t|  sum\t|");
    $monitor("|   %3d \t|  %2d \t|  %2d \t|  %2d \t|  %2d \t|", jj, i_a, i_b, c_in, {c_out, o_sum});
    for (int ii = 0; ii < 512; ii++) begin
      i_a = ii[3:0];
      i_b = ii[7:4];
      c_in = ii[8];
      #1;
  

      assert ({c_out, o_sum} == (jj[3:0] + jj[7:4] + jj[8]))
        else $error("ERROR!!!!\t\t\t a\t %0d b\t %0d c_in\t %0d sum \t %0d \t carry \t %0d\r\n Expected: %0d", 
            i_a, i_b, c_in, {c_out, o_sum}, c_out, (jj[3:0] + jj[7:4] + jj[8]));
    // $display ("Display:\t\t\t a\t %0b\t b\t %0b\t c_in\t %0b\t sum \t %0b \t carry \t %0b", i_a, i_b, c_in, o_sum, c_out);

            jj++;
    end
    // $display ("sum \t %0b \t carry \t %0b", o_sum, c_out);
  end

  initial begin
    $dumpfile("dump.vcd");
    $dumpvars(2, four_bits_tb);
    $dumpvars(2, four_bits);

    #600
    $finish;
  end

endmodule

module carry_sel_tb;

// Parameters

//Ports
reg [12 - 1 : 0] i_a;
reg [12 - 1 : 0] i_b;
reg  c_in;
wire [12 : 0] o_sum;
wire  c_out;
event start;
wire [12 : 0 ] s_out;

// assign s_out = {c_out, o_sum};
assign o_sum [12] = c_out;
carry_sel  carry_sel_inst (
  .i_a(i_a),
  .i_b(i_b),
  .c_in(c_in),
  .o_sum(o_sum),
  .c_out(c_out)
);
bit [24:0] jj;

//always #5  clk = ! clk ;
  initial begin
    i_a = 'x;
    i_b = 'x;
    c_in = 'x;
    #1;
    $display("|\t\tCount\t|\t a\t\t|\t b\t\t|\t c\t|\t sum\t|");
    $display("|   %9d \t|  %4h \t|  %4h \t|  %2h \t|  %5h \t|", jj, i_a, i_b, c_in, {c_out, o_sum});
    for (int ii = 0; ii < 2**25; ii++) begin
      i_a = ii[11:0];
      i_b = ii[23:12];
      c_in = ii[24];
      #1;
  
      assert ({c_out, o_sum} == (jj[11:0] + jj[23:12] + jj[24]))
        else $fatal("ERROR!!!!\t\t\t a\t %0d b\t %0d c_in\t %0d sum \t %0d \t carry \t %0d\n Expected: %0d", 
            i_a, i_b, c_in, {c_out, o_sum}, c_out, (jj[11:0] + jj[23:12] + jj[24]));

      if (jj % 234561 == 0) 
      $display("|   %9d \t|  %4d \t|  %4d \t|  %2d \t|  %5d \t|", jj, i_a, i_b, c_in, {c_out, o_sum});
      jj++;
    end
    -> start;

    /////////////////////////////////////////////////
    //Phase 2 start
    // No need to assert the output as it was already
    // tested in Phase 1
    /////////////////////////////////////////////////

    ///////////////////////////////////////////////
    // 0x0 (a) + 0x0 (b) + 0x0 (c_in) = 0x0 ({c_out, s})
    //////////////////////////////////////////////
    i_a = '0;
    i_b = '0;
    c_in = '0;
    #5;

    ///////////////////////////////////////////////
    // 0xFFF (a) + 0x0 (b) + 0x0 (c_in) = 0x0FFF ({c_out, s})
    //////////////////////////////////////////////
    i_a = 'hFFF;
    i_b = '0;
    c_in = '0;
    #5;
    
    ///////////////////////////////////////////////
    // 0xFFF (a) + 0x0 (b) + 0x1 (c_in) = 0x1000 ({c_out, s})
    //////////////////////////////////////////////
    i_a = 'hFFF;
    i_b = '0;
    c_in = 'b1;
    #5;
    
    ///////////////////////////////////////////////
    // 0x420 (a) + 0xFFF (b) + 0x1 (c_in) = 0x142F ({c_out, s})
    //////////////////////////////////////////////
    i_a = 'h420;
    i_b = 'hFFF;
    c_in = '0;
    #5;
    
    ///////////////////////////////////////////////
    // 0xFFF (a) + 0xFFF (b) + 0x0 (c_in) = 0x1FFE({c_out, s})
    //////////////////////////////////////////////
    i_a = 'hFFF;
    i_b = 'hFFF;
    c_in = '0;
    #5;

    
    ///////////////////////////////////////////////
    // 0xFFF (a) + 0xFFF (b) + 0x1 (c_in) = 0x1FFF({c_out, s})
    //////////////////////////////////////////////
    i_a = 'hFFF;
    i_b = 'hFFF;
    c_in = 'b1;
    #5;


  end

  initial begin
    $dumpfile("dump.vcd");
    $dumpvars(2, carry_sel_tb);
    $dumpvars(2, carry_sel);
    $dumpoff;

    @(start);
    $dumpon;

    #400
    $finish;
  end

endmodule

// $finish called at time : 40 ms : File "/home/lolwut/utd/eect/7325/adder_tb.sv" Line 99
// run: Time (s): cpu = 00:00:06 ; elapsed = 00:02:22 . Memory (MB): peak = 10780.129 ; gain = 17.066 ; free physical = 21481 ; free virtual = 35085
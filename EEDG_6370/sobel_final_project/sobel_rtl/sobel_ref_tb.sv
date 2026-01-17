
module sobel_tb;

  // Parameters
  localparam  DATAWIDTH = 32;
  localparam  MEMORYWIDTH = 8;
  localparam  ADDRWIDTH = 6;

  //Ports
  reg  CLK;
  reg  ARESETN;
  reg [DATAWIDTH - 1 : 0] S_AXIS_TDATA;
  reg  S_AXIS_TLAST;
  reg  S_AXIS_TVALID;
  wire  S_AXIS_TREADY;
  wire    [DATAWIDTH - 1 : 0] M_AXIS_TDATA;
  wire  M_AXIS_TLAST;
  wire  M_AXIS_TVALID;
  reg  M_AXIS_TREADY;
  wire                 [3 : 0] LED;
  wire                 [2 : 0] RGB0;
  wire                 [2 : 0] RGB1;

  sobel_ref_top # (
    .DATAWIDTH(DATAWIDTH),
    .MEMORYWIDTH(MEMORYWIDTH),
    .ADDRWIDTH(ADDRWIDTH)
  )
  sobel_ref_top_inst (
    .CLK(CLK),
    .ARESETN(ARESETN),
    .S_AXIS_TDATA(S_AXIS_TDATA),
    .S_AXIS_TLAST(S_AXIS_TLAST),
    .S_AXIS_TVALID(S_AXIS_TVALID),
    .S_AXIS_TREADY(S_AXIS_TREADY),
    .M_AXIS_TDATA(M_AXIS_TDATA),
    .M_AXIS_TLAST(M_AXIS_TLAST),
    .M_AXIS_TVALID(M_AXIS_TVALID),
    .M_AXIS_TREADY(M_AXIS_TREADY),
    .LED(LED),
    .RGB0(RGB0),
    .RGB1(RGB1)
  );

  always #5  CLK = ! CLK ;
  wire clk = CLK;

  initial begin
    CLK = 0;
    $dumpfile("dump.vcd");
    $dumpvars(2, sobel_tb);
    $dumpvars(2, sobel_ref_top);
    #6000;
    $finish;
  end

  initial begin
    S_AXIS_TDATA = 'hDEADBEEF;
    S_AXIS_TVALID = 'b0;
    ARESETN = '0;
    S_AXIS_TLAST = 'b0;

    M_AXIS_TREADY = 'b0;

    @ (negedge clk);
    @ (negedge clk);
    ARESETN = 1'b1;
    @ (negedge clk);
    for (int kk  = 0; kk < 2; kk++) begin
      @( negedge clk);
      S_AXIS_TLAST = 1'b0;
      
      for (int ii = 0; ii < 31; ii++) begin
        @(negedge clk);
        S_AXIS_TVALID = 1'b1;
        S_AXIS_TDATA = 'h0;
        M_AXIS_TREADY = 1'b1;
      end
      @(negedge clk);
      S_AXIS_TDATA = 'h0;
      S_AXIS_TLAST = 1'b1;
      // @(negedge clk);

      for (int ii = 0; ii < 20; ii++) begin
        @(negedge clk);
        S_AXIS_TVALID = 1'b0;
        M_AXIS_TREADY = 1'b1;
        S_AXIS_TDATA = 'hDEADBEEF;
        S_AXIS_TLAST = 1'b0;
      end
      
    end
    for (int kk  = 0; kk < 4; kk++) begin
      @( negedge clk);
      S_AXIS_TLAST = 1'b0;
      
      for (int ii = 0; ii < 31; ii++) begin
        @(negedge clk);
        S_AXIS_TVALID = 1'b1;
        if ((ii < 5) || (ii > 27)) begin
        S_AXIS_TDATA = 'h0;
        end else begin
        S_AXIS_TDATA = 'h55FFFFFF;
        end
        M_AXIS_TREADY = 1'b1;
      end
      @(negedge clk);
      S_AXIS_TDATA = 'h0;
      S_AXIS_TLAST = 1'b1;
      // @(negedge clk);

      for (int ii = 0; ii < 20; ii++) begin
        @(negedge clk);
        S_AXIS_TVALID = 1'b0;
        M_AXIS_TREADY = 1'b1;
        S_AXIS_TDATA = 'hDEADBEEF;
        S_AXIS_TLAST = 1'b0;
      end
      
    end
  end
endmodule

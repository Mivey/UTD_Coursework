module sobel_ref_top #(
  parameter DATAWIDTH = 32,
  parameter MEMORYWIDTH = 8,
  parameter ADDRWIDTH = 12
) (
  input wire    CLK,
  input wire    ARESETN,

  input   wire    [DATAWIDTH - 1 : 0] S_AXIS_TDATA,
  input   wire                        S_AXIS_TLAST,
  input   wire                        S_AXIS_TVALID,
  output  wire                        S_AXIS_TREADY,

  output  wire    [DATAWIDTH - 1 : 0] M_AXIS_TDATA,
  output  wire                        M_AXIS_TLAST,
  output  wire                        M_AXIS_TVALID,
  input   wire                        M_AXIS_TREADY,

  output wire                 [3 : 0] LED,
  output wire                 [2 : 0] RGB0,
  output wire                 [2 : 0] RGB1
  );


sobel # (
    .DATAWIDTH(DATAWIDTH),
    .MEMORYWIDTH(MEMORYWIDTH),
    .ADDRWIDTH(ADDRWIDTH)
  )
  sobel_inst (
    .clk(CLK),
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

endmodule



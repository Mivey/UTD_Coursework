module bram #(
  parameter DATAWIDTH = 8,
  parameter ADDRWIDTH = 12
) (
  input   logic clk,
  // input   logic en,
  input   logic w_en,
  input   logic r_en,
  input   logic [ADDRWIDTH - 1 : 0] wptr,
  input   logic [ADDRWIDTH - 1 : 0] rptr,
  input   logic [DATAWIDTH - 1 : 0] data_in,
  output  logic [DATAWIDTH - 1 : 0] data_out
);

logic [DATAWIDTH - 1 : 0] mem [2**ADDRWIDTH - 1];

initial begin
  for (int i = 0; i < 2 ** ADDRWIDTH - 1; i++) begin
    mem[i] = 'b0;
  end
end

always_ff @ (posedge clk) begin
  if (w_en) begin
    mem [wptr] <= data_in;
  end 

  if (r_en) begin
    data_out <= mem[rptr];
  end
end
 
endmodule
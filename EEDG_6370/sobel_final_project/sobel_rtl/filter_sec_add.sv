module filter_sec_add #(
  parameter PIXEL = 3,
  parameter ROW_LOOP = 3,
  parameter DATAWIDTH = 8
) (
  input wire clk,
  input wire ARESETN,
  input wire [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_sum_x,
  input wire [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_sum_y,
  output logic [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_sum,
  input wire i_busy,
  input wire i_strobe,
  output logic o_busy,
  output logic o_strobe,
  input wire in_tlast,
  output logic out_tlast


);

logic [DATAWIDTH * 2 - 1 : 0] sumX [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] sumY [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] sum [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] r_sum [PIXEL];

// logic [DATAWIDTH - 1 : 0] data_out [PIXEL][ROW_LOOP];

genvar i;
genvar j;
generate for (i = 0; i < PIXEL; i++) begin 
  assign packed_sum[DATAWIDTH * 2 * i +: DATAWIDTH * 2] = sum[i];
  assign sumX[i] = packed_sum_x[DATAWIDTH * 2 * i +: DATAWIDTH * 2];
  assign sumY[i] = packed_sum_y[DATAWIDTH * 2 * i +: DATAWIDTH * 2];
end
endgenerate

logic r_strobe;
logic r_tlast;

  always_ff @(posedge clk) begin
    if (!ARESETN) begin
      o_busy <= 1'b1;
      o_strobe <= 1'b0;
      r_strobe <= 1'b0;
      r_tlast <= 1'b0;

    end else begin

      //Pipelined output

      if (!i_busy) begin
        o_busy <= 1'b0;
        r_strobe <= 1'b0;

        if (!r_strobe) begin
          o_strobe <= i_strobe;

          for (int k = 0; k < PIXEL; k++) begin
            sum[k] <= sumX[k] + sumY[k];
          end

          out_tlast <= in_tlast;
        end else begin
          for (int k = 0; k < PIXEL; k++) begin
            sum[k] <= r_sum[k];
          end
          out_tlast <= r_tlast;

          o_strobe <= 1'b1;
        end
      end else begin
        if (!o_strobe) begin
          o_strobe <= i_strobe;
          o_busy <= 1'b0;
          r_strobe <= 1'b0;

          for (int k = 0; k < PIXEL; k++) begin
            sum[k] <= sumX[k] + sumY[k];
          end
          out_tlast <= in_tlast;
        end else begin
          if ( i_strobe && (!o_busy)) begin
            r_strobe <= i_strobe && o_strobe;
            o_busy <= i_strobe && o_strobe;
          end
        end
      end
      if (o_busy) begin
        for (int k = 0; k < PIXEL; k++) begin
            r_sum[k] <= sumX[k] + sumY[k];
        end
        r_tlast <= in_tlast;
      end
    end
  end 
endmodule

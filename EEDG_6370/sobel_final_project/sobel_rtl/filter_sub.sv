module filter_sub #(
  parameter PIXEL = 3,
  parameter ROW_LOOP = 3,
  parameter DATAWIDTH = 8
) (
  input wire clk,
  input wire ARESETN,
  input wire [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_out_x_0,
  input wire [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_out_x_2,
  input wire [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_out_y_0,
  input wire [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_out_y_2,
  output logic [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_sum_x,
  output logic [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_sum_y,
  input wire i_busy,
  input wire i_strobe,
  output logic o_busy,
  output logic o_strobe,
  input wire in_tlast,
  output logic out_tlast


);

logic [DATAWIDTH * 2 - 1 : 0] p_0_sum_x [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] p_2_sum_x [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] p_0_sum_y [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] p_2_sum_y [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] sumX [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] sumY [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] r_sumX [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] r_sumY [PIXEL];

// logic [DATAWIDTH - 1 : 0] data_out [PIXEL][ROW_LOOP];

genvar i;
genvar j;
generate for (i = 0; i < PIXEL; i++) begin 

  assign p_0_sum_x [i] = packed_out_x_0[DATAWIDTH * 2 * (i + 0) +: DATAWIDTH * 2];
  assign p_2_sum_x [i] = packed_out_x_2[DATAWIDTH * 2 * (i + 0) +: DATAWIDTH * 2];
  assign p_0_sum_y [i] = packed_out_y_0[DATAWIDTH * 2 * (i + 0) +: DATAWIDTH * 2];
  assign p_2_sum_y [i] = packed_out_y_2[DATAWIDTH * 2 * (i + 0) +: DATAWIDTH * 2];

  assign packed_sum_x[DATAWIDTH * 2 * i +: DATAWIDTH * 2] = sumX[i];
  assign packed_sum_y[DATAWIDTH * 2 * i +: DATAWIDTH * 2] = sumY[i];
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
            sumX[k] <= (p_2_sum_x[k] >= p_0_sum_x[k]) ? 'b0 : p_0_sum_x[k] - p_2_sum_x[k];
            sumY[k] <= (p_2_sum_y[k] >= p_0_sum_y[k]) ? 'b0 : p_0_sum_y[k] - p_2_sum_y[k];
          end

          out_tlast <= in_tlast;
        end else begin
          for (int k = 0; k < PIXEL; k++) begin
            sumX[k] <= r_sumX[k];
            sumY[k] <= r_sumY[k];
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
            sumX[k] <= (p_2_sum_x[k] >= p_0_sum_x[k]) ? 'b0 : p_0_sum_x[k] - p_2_sum_x[k];
            sumY[k] <= (p_2_sum_y[k] >= p_0_sum_y[k]) ? 'b0 : p_0_sum_y[k] - p_2_sum_y[k];
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
            r_sumX[k] <= (p_2_sum_x[k] >= p_0_sum_x[k]) ? 'b0 : p_0_sum_x[k] - p_2_sum_x[k];
            r_sumY[k] <= (p_2_sum_y[k] >= p_0_sum_y[k]) ? 'b0 : p_0_sum_y[k] - p_2_sum_y[k];
        end
        r_tlast <= in_tlast;
      end
    end
  end 
endmodule
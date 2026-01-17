module filter_add #(
  parameter PIXEL = 3,
  parameter ROW_LOOP = 3,
  parameter DATAWIDTH = 8
) (
  input wire clk,
  input wire ARESETN,
  input wire [DATAWIDTH * PIXEL * ROW_LOOP - 1 : 0] packed_in_0,
  input wire [DATAWIDTH * PIXEL * ROW_LOOP - 1 : 0] packed_in_1,
  input wire [DATAWIDTH * PIXEL * ROW_LOOP - 1 : 0] packed_in_2,
  output logic [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_out_x_0,
  output logic [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_out_x_2,
  output logic [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_out_y_0,
  output logic [DATAWIDTH * 2 * PIXEL - 1 : 0] packed_out_y_2,
  input wire i_busy,
  input wire i_strobe,
  output logic o_busy,
  output logic o_strobe,
  input wire in_tlast,
  output logic out_tlast


);
logic [DATAWIDTH - 1 : 0] data_out_0 [PIXEL][ROW_LOOP];
logic [DATAWIDTH - 1 : 0] data_out_1 [PIXEL][ROW_LOOP];
logic [DATAWIDTH - 1 : 0] data_out_2 [PIXEL][ROW_LOOP];
logic [DATAWIDTH * 2 - 1 : 0] p_0_sum_x [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] p_2_sum_x [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] p_0_sum_y [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] p_2_sum_y [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] r_p_0_sum_x [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] r_p_2_sum_x [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] r_p_0_sum_y [PIXEL];
logic [DATAWIDTH * 2 - 1 : 0] r_p_2_sum_y [PIXEL];
// logic [DATAWIDTH - 1 : 0] data_out [PIXEL][ROW_LOOP];

genvar i;
genvar j;
generate for (i = 0; i < PIXEL; i++) begin 
  for (j = 0; j < ROW_LOOP; j++) begin
    assign data_out_0 [i][j] = packed_in_0[DATAWIDTH * (ROW_LOOP * i + (j + 0)) +: DATAWIDTH];
    assign data_out_1 [i][j] = packed_in_1[DATAWIDTH * (ROW_LOOP * i + (j + 0)) +: DATAWIDTH];
    assign data_out_2 [i][j] = packed_in_2[DATAWIDTH * (ROW_LOOP * i + (j + 0)) +: DATAWIDTH];
    // assign packed_out [DATAWIDTH * (ROW_LOOP * i + (j + 1)) +: 8] = data_out[i][j];
  end

  assign packed_out_x_0[DATAWIDTH * 2 * (i + 0) +: DATAWIDTH * 2] = p_0_sum_x [i];
  assign packed_out_x_2[DATAWIDTH * 2 * (i + 0) +: DATAWIDTH * 2] = p_2_sum_x [i];
  assign packed_out_y_0[DATAWIDTH * 2 * (i + 0) +: DATAWIDTH * 2] = p_0_sum_y [i];
  assign packed_out_y_2[DATAWIDTH * 2 * (i + 0) +: DATAWIDTH * 2] = p_2_sum_y [i];
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
            p_2_sum_x[k] <= data_out_0[k][2] + (data_out_0[k][1] << 1) + data_out_0[k][0];
            p_0_sum_x[k] <= data_out_2[k][2] + (data_out_2[k][1] << 1) + data_out_2[k][0];
            p_0_sum_y[k] <= data_out_0[k][0] + (data_out_1[k][0] << 1) + data_out_2[k][0];
            p_2_sum_y[k] <= data_out_0[k][2] + (data_out_1[k][2] << 1) + data_out_2[k][2];
          end

          out_tlast <= in_tlast;
        end else begin
          for (int k = 0; k < PIXEL; k++) begin
            p_2_sum_x[k] <= r_p_2_sum_x[k];
            p_0_sum_x[k] <= r_p_0_sum_x[k];
            p_0_sum_y[k] <= r_p_2_sum_y[k];
            p_2_sum_y[k] <= r_p_2_sum_y[k];
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
            p_2_sum_x[k] <= data_out_0[k][2] + (data_out_0[k][1] << 1) + data_out_0[k][0];
            p_0_sum_x[k] <= data_out_2[k][2] + (data_out_2[k][1] << 1) + data_out_2[k][0];
            p_0_sum_y[k] <= data_out_0[k][0] + (data_out_1[k][0] << 1) + data_out_2[k][0];
            p_2_sum_y[k] <= data_out_0[k][2] + (data_out_1[k][2] << 1) + data_out_2[k][2];
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
          r_p_2_sum_x[k] <= data_out_0[k][2] + (data_out_0[k][1] << 1) + data_out_0[k][0];
          r_p_0_sum_x[k] <= data_out_2[k][2] + (data_out_2[k][1] << 1) + data_out_2[k][0];
          r_p_0_sum_y[k] <= data_out_0[k][0] + (data_out_1[k][0] << 1) + data_out_2[k][0];
          r_p_2_sum_y[k] <= data_out_0[k][2] + (data_out_1[k][2] << 1) + data_out_2[k][2];
        end
        r_tlast <= in_tlast;
      end
    end
  end

  
endmodule
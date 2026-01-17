module buffer_2d #(
  parameter PIXEL = 3,
  parameter ROW_LOOP = 3,
  parameter DATAWIDTH = 8
) (
  input wire clk,
  input wire ARESETN,
  input wire [DATAWIDTH * PIXEL * ROW_LOOP - 1 : 0] packed_in,
  output logic [DATAWIDTH * PIXEL * ROW_LOOP - 1 : 0] packed_out,
  input wire i_busy,
  input wire i_strobe,
  output logic o_busy,
  output logic o_strobe,
  input wire in_tlast,
  output logic out_tlast


);
logic [DATAWIDTH - 1 : 0] data_in [PIXEL][ROW_LOOP];
logic [DATAWIDTH - 1 : 0] data_out [PIXEL][ROW_LOOP];

genvar i;
genvar j;
generate for (i = 0; i < PIXEL; i++) begin 
  for (j = 0; j < ROW_LOOP; j++) begin
    assign data_in [i][j] = packed_in[DATAWIDTH * (ROW_LOOP * i + (j + 0)) +: 8];
    assign packed_out [DATAWIDTH * (ROW_LOOP * i + (j + 0)) +: 8] = data_out[i][j];
  end
end
endgenerate

logic r_strobe;
logic r_tlast;
logic [DATAWIDTH - 1 : 0] r_data_out [PIXEL][ROW_LOOP];

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
            data_out[k][0] <= data_in[k][0];
            data_out[k][1] <= data_in[k][1];
            data_out[k][2] <= data_in[k][2];
          end

          out_tlast <= in_tlast;
        end else begin
          for (int k = 0; k < PIXEL; k++) begin
            data_out[k][0] <= r_data_out[k][0];
            data_out[k][1] <= r_data_out[k][1];
            data_out[k][2] <= r_data_out[k][2];
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
            data_out[k][0] <= data_in[k][0];
            data_out[k][1] <= data_in[k][1];
            data_out[k][2] <= data_in[k][2];
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
          r_data_out[k][0] <= data_in[k][0];
          r_data_out[k][1] <= data_in[k][1];
          r_data_out[k][2] <= data_in[k][2];
        end
        r_tlast <= in_tlast;
      end
    end
  end

  
endmodule

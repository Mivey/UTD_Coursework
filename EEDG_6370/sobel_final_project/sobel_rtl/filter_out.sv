module filter_out #(
  parameter PIXEL = 3,
  parameter ROW_LOOP = 3,
  parameter MEMORYWIDTH = 8,
  parameter DATAWIDTH = 32
) (
  input wire clk,
  input wire ARESETN,
  input wire [MEMORYWIDTH * 2 * PIXEL - 1 : 0] packed_sum,
  output logic [DATAWIDTH - 1 : 0] M_AXIS_TDATA,
  input wire i_busy,
  input wire i_strobe,
  output logic o_busy,
  output logic o_strobe,
  input wire in_tlast,
  output logic out_tlast


);

logic [MEMORYWIDTH * 2 - 1 : 0] sum [PIXEL];
logic [DATAWIDTH - 1 : 0] r_m_axis_tdata;

// logic [MEMORYWIDTH - 1 : 0] data_out [PIXEL][ROW_LOOP];

genvar i;
generate for (i = 0; i < PIXEL; i++) begin 
  assign sum[i] = packed_sum[MEMORYWIDTH * 2 * i +: MEMORYWIDTH * 2];
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

      M_AXIS_TDATA[MEMORYWIDTH - 1 : 0] <= 'b0;
      if (!i_busy) begin
        o_busy <= 1'b0;
        r_strobe <= 1'b0;

        if (!r_strobe) begin
          o_strobe <= i_strobe;

          for (int k = 0; k < PIXEL; k++) begin
            M_AXIS_TDATA[(k + 1) * MEMORYWIDTH +: MEMORYWIDTH] <= (sum[k] > 255) ? 255 : sum[k];
          end

          out_tlast <= in_tlast;
        end else begin
          for (int k = 0; k < PIXEL; k++) begin
            M_AXIS_TDATA[(k + 1) * MEMORYWIDTH +: MEMORYWIDTH] <= r_m_axis_tdata[(k + 1) * MEMORYWIDTH +: MEMORYWIDTH];
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
            M_AXIS_TDATA[(k + 1) * MEMORYWIDTH +: MEMORYWIDTH] <= (sum[k] > 255) ? 255 : sum[k];
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
            r_m_axis_tdata[(k + 1) * MEMORYWIDTH +: MEMORYWIDTH] <= (sum[k] > 255) ? 255 : sum[k];
        end
        r_tlast <= in_tlast;
      end
    end
  end 
endmodule

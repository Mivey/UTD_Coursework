`default_nettype none

/////////////// Mirror Carry////////////////////////
module m_carry (
    input  wire a,
    input  wire b,
    input  wire c,
    output wire c_out_n
);
  assign c_out_n = ~((a & b) | ((a | b) & c));
endmodule

//////////////// Mirror Sum ///////////////////////////
module m_sum (
    input  wire a,
    input  wire b,
    input  wire c_in,
    input  wire c_out_n,
    output wire s_out
);
  assign s_out = ~((c_in & a & b) | ((c_in | a | b) & c_out_n));
endmodule

//////////////// Four Bit Ripple Carry Adder //////////////////////
module four_bits (
    input wire [3:0] i_a,
    input wire [3:0] i_b,
    input wire c_in,
    output wire c_out,
    output wire [3:0] o_sum
);
  wire [3:0] a, b, sum;
  wire [4:0] carry, n_carry;
  genvar i;

  generate
    for (i = 0; i < 4; i = i + 1) begin
      m_sum m_sum_0 (
          .a(a[i]),
          .b(b[i]),
          .c_in(carry[i]),
          .c_out_n(carry[i+1]),
          .s_out(sum[i])
      );

      m_carry m_carry_0 (
          .a(a[i]),
          .b(b[i]),
          .c(carry[i]),
          .c_out_n(carry[i+1])
      );


      ////// not gates for even/odd bits /////
      if (i % 2 == 0) begin
        assign a[i] = i_a[i];
        assign b[i] = i_b[i];
        assign o_sum[i] = ~sum[i];
      end else begin
        assign a[i] = ~i_a[i];
        assign b[i] = ~i_b[i];
        assign o_sum[i] = sum[i];
      end
    end
  endgenerate

  assign c_out = carry[4];
  assign carry[0] = c_in;
endmodule

////////////////// Carry Select Adder ////////////////////////
module carry_sel (
    input wire [12 - 1 : 0] i_a,
    input wire [12 - 1 : 0] i_b,
    input wire c_in,
    output wire [12 - 1 : 0] o_sum,
    output wire c_out
);

  wire cs_3, cs_l, cs_u, c_out_l, c_out_u;
  wire [11:4] sum_lb, sum_ub;
  wire ub_cs;
  assign c_out = (ub_cs) ? c_out_u : c_out_l;   //bit 12, or c_out mux

  ////////// Std ripple carry for bits 3:0
  four_bits four_bits_3_0 (
      .i_a  (i_a[3:0]),
      .i_b  (i_b[3:0]),
      .c_in (c_in),
      .c_out(cs_3),
      .o_sum(o_sum[3:0])
  );

  //////////// Bits 7:4 with carry-in/no carry-in ////////////
  four_bits four_bits_7_4_L (
      .i_a  (i_a[7:4]),
      .i_b  (i_b[7:4]),
      .c_in (1'b0),
      .c_out(cs_l),
      .o_sum(sum_lb[7:4])
  );
  four_bits four_bits_7_4_U (
      .i_a  (i_a[7:4]),
      .i_b  (i_b[7:4]),
      .c_in (1'b1),
      .c_out(cs_u),
      .o_sum(sum_ub[7:4])
  );

  ////// Bits 11:8 with carry-in/no carry-in //////////////
  four_bits four_bits_11_8_L (
      .i_a  (i_a[11:8]),
      .i_b  (i_b[11:8]),
      .c_in (1'b0),
      .c_out(c_out_l),
      .o_sum(sum_lb[11:8])
  );
  four_bits four_bits_11_8_U (
      .i_a  (i_a[11:8]),
      .i_b  (i_b[11:8]),
      .c_in (1'b1),
      .c_out(c_out_u),
      .o_sum(sum_ub[11:8])
  );

  assign ub_cs = (cs_3 & cs_u) | cs_l;    // input logic for 11:8 output assignment mux
  assign o_sum[7 : 4] = cs_3 ? sum_ub[7:4] : sum_lb[7:4];   // 7:4 mux 
  assign o_sum[11 : 8] = ub_cs ? sum_ub[11:8] : sum_lb[11:8];   // 11:8 mux

endmodule

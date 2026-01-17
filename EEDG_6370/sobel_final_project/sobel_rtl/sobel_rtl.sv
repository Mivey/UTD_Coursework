`default_nettype none 

module sobel #(
    parameter DATAWIDTH   = 32,
    parameter MEMORYWIDTH = 8,
    parameter ADDRWIDTH   = 11
) (
    input wire clk,
    input wire ARESETN,

    input  wire [DATAWIDTH - 1 : 0]   S_AXIS_TDATA,
    input  wire                       S_AXIS_TLAST,
    input  wire                       S_AXIS_TVALID,
    output logic                      S_AXIS_TREADY,

    output logic  [DATAWIDTH - 1 : 0] M_AXIS_TDATA,
    output logic                      M_AXIS_TLAST,
    output logic                      M_AXIS_TVALID,
    input  wire                       M_AXIS_TREADY,

    output logic [3 : 0] LED,
    output logic [2 : 0] RGB0,
    output logic [2 : 0] RGB1
);

  localparam int PIXEL = 3;
  localparam int ROW_LOOP = 3;
  localparam int PIPELINE_STAGE = 7;

  logic [ROW_LOOP - 1 : 0] w_en;  // One row in each of the three pixels will be enabled.
  logic [ROW_LOOP - 1 : 0] comb_w_en;  // combinitorial block says which w_en is active.
                                        // Seq. Block controls w_en by passing this value.
  logic r_en;  // we will read all data from all pixels and rows
  logic [MEMORYWIDTH - 1 : 0] bram_in[PIXEL]; // Each pixel gets its own 8 bits, so those are enabled 
  logic [MEMORYWIDTH - 1 : 0] bram_out[PIXEL][ROW_LOOP];  // output data from all rows/pixels
  logic [MEMORYWIDTH - 1 : 0] m_bram_out [PIXEL][ROW_LOOP];

  logic [ADDRWIDTH - 1 : 0] col_read_ptr;  // main read pointer. 
  logic [ADDRWIDTH - 1 : 0] col_write_ptr; // main write pointer. adds next element to mem array for given row
  logic [1 : 0] row_write_ptr;  // controls the row we are currently writing to

  //=============================== MEMEORY GENERATE ====================================
  genvar t, s;

  generate

    for (t = 0; t < PIXEL; t = t + 1) begin : PIXEL_LOOP
      for (s = 0; s < ROW_LOOP; s = s + 1) begin : ROW_LOOPING
        bram #(
            .DATAWIDTH(MEMORYWIDTH),
            .ADDRWIDTH(ADDRWIDTH)
        ) bram_inst (
            .clk(clk),
            .w_en(w_en[s]),
            .r_en(r_en),
            .wptr(col_write_ptr),
            .rptr(col_read_ptr),
            .data_in(bram_in[t]),  // we tie row loop together, separate pixel
            .data_out(bram_out[t][s])
        );
      end

    end
  endgenerate
  //==========================================================================================


  // ======================== States for the FSM. ==============================================
  typedef enum {
    INIT,           // reset and wait for ready/valid
    TRANSMIT,       // start pipeline, start collecting data into bram
    STALL,          // case where s_rv goes low due to stall on DMA from either side
    COMPLETE,        // s_l == 1, wait for m_l == 1 and set flag to change which block of bram to write to
    XXX             // invalid state
  } fsm_state_t;
  fsm_state_t c_state;  // output of the FF
  fsm_state_t n_state;  // FF input
  //===================================================================================================

  logic [1 : 0] row_top;
  logic [1 : 0] row_mid;
  logic [1 : 0] row_end;
  logic shuffle_flag;

  logic [PIPELINE_STAGE - 1 : 0] i_busy;
  logic [PIPELINE_STAGE - 1 : 0] o_strobe;
  logic [PIPELINE_STAGE - 1 : 0] i_tlast;

  logic [PIPELINE_STAGE - 1 : 0] o_busy;
  logic [PIPELINE_STAGE - 1 : 0] i_strobe;
  logic [PIPELINE_STAGE - 1 : 0] o_tlast;

 /*                                       _______________________________
             i_busy_0      o_busy_1 <----|   block 1                    |<---- i_busy_1      o_busy_2
             o_strobe_0  i_strobe_1  --->|                              |---> o_strobe_1     i_strobe_2
             o_tlast_0     i_tlast_1 --->|______________________________|---> o_tlast_1      i_tlast_2
  */

  logic s_p_busy_o; // (S)tart (P)ipeline busy
  logic s_p_strobe_i; // (S)tart (P)ipeline strobe
  logic s_p_tlast_i; // (S)tart (P)ipeline tlast

  logic e_p_busy_i; // (E)nd (P)ipeline busy
  logic e_p_strobe_o; // (E)nd (P)ipeline strobe
  logic e_p_tlast_o; // (E)nd (P)ipeline tlast
  
  wire [MEMORYWIDTH * ROW_LOOP * PIXEL - 1 : 0] top_to_stage_0;
  wire [MEMORYWIDTH * ROW_LOOP * PIXEL - 1 : 0] top_to_stage_1;
  wire [MEMORYWIDTH * ROW_LOOP * PIXEL - 1 : 0] top_to_stage_2;
  wire [MEMORYWIDTH * ROW_LOOP * PIXEL - 1 : 0] top_to_stage_3;

  wire [MEMORYWIDTH * 2 * PIXEL - 1 : 0] packed_out_x_0;
  wire [MEMORYWIDTH * 2 * PIXEL - 1 : 0] packed_out_x_2;
  wire [MEMORYWIDTH * 2 * PIXEL - 1 : 0] packed_out_y_0;
  wire [MEMORYWIDTH * 2 * PIXEL - 1 : 0] packed_out_y_2;

  wire [MEMORYWIDTH * 2 * PIXEL - 1 : 0] packed_sum_x;
  wire [MEMORYWIDTH * 2 * PIXEL - 1 : 0] packed_sum_y;

  wire [MEMORYWIDTH * 2 * PIXEL - 1 : 0] packed_sum;

  genvar ii;
  genvar jj;
  generate for (ii = 0; ii < PIXEL; ii++) begin 
    for (jj = 0; jj < ROW_LOOP; jj++) begin
      // assign data_in [ii][jj] = packed_in[DATAWIDTH * (ROW_LOOP * ii + (jj + 1)) +: 8];
      assign top_to_stage_0 [MEMORYWIDTH * (ROW_LOOP * ii + (jj + 0)) +: 8] = m_bram_out[ii][jj];
    end
    assign m_bram_out[ii][0] = bram_out[ii][row_top];
    assign m_bram_out[ii][1] = bram_out[ii][row_mid];
    assign m_bram_out[ii][2] = bram_out[ii][row_end];
  end
  endgenerate


  //Pipelined output

  genvar kk;
  generate for (kk = 0; kk < PIPELINE_STAGE - 1; kk++) begin
    assign i_busy[kk] = o_busy[kk + 1];
    assign i_strobe[kk + 1] = o_strobe[kk];
    assign i_tlast[kk  + 1] = o_tlast[kk];
  end
  endgenerate

  assign s_p_busy_o = o_busy[0];
  assign i_strobe[0] = s_p_strobe_i;
  assign i_tlast[0] = s_p_tlast_i;

  assign i_busy[6] = e_p_busy_i;
  assign e_p_strobe_o = o_strobe[6];
  assign e_p_tlast_o = o_tlast[6];  

  assign s_p_tlast_i = S_AXIS_TLAST;
  assign M_AXIS_TLAST = e_p_tlast_o;
  assign s_p_strobe_i = S_AXIS_TVALID;
  assign e_p_busy_i = ! M_AXIS_TREADY;
  assign M_AXIS_TVALID = e_p_strobe_o;


  //Stage 0
  buffer_2d # (
    .PIXEL(PIXEL),
    .ROW_LOOP(ROW_LOOP),
    .DATAWIDTH(MEMORYWIDTH)
  )
  buffer_2d_inst_0 (
    .clk(clk),
    .ARESETN(ARESETN),
    .packed_in(top_to_stage_0),
    .packed_out(top_to_stage_1),
    .i_busy(i_busy[0]),
    .i_strobe(i_strobe[0]),
    .o_busy(o_busy[0]),
    .o_strobe(o_strobe[0]),
    .in_tlast(i_tlast[0]),
    .out_tlast(o_tlast[0])
  );
  
  //Stage 1
  buffer_2d # (
    .PIXEL(PIXEL),
    .ROW_LOOP(ROW_LOOP),
    .DATAWIDTH(MEMORYWIDTH)
  )
  buffer_2d_inst (
    .clk(clk),
    .ARESETN(ARESETN),
    .packed_in(top_to_stage_1),
    .packed_out(top_to_stage_2),
    .i_busy(i_busy[1]),
    .i_strobe(i_strobe[1]),
    .o_busy(o_busy[1]),
    .o_strobe(o_strobe[1]),
    .in_tlast(i_tlast[1]),
    .out_tlast(o_tlast[1])
  );
  
  //Stage 2
  buffer_2d # (
    .PIXEL(PIXEL),
    .ROW_LOOP(ROW_LOOP),
    .DATAWIDTH(MEMORYWIDTH)
  )
  buffer_2d_inst_2 (
    .clk(clk),
    .ARESETN(ARESETN),
    .packed_in(top_to_stage_2),
    .packed_out(top_to_stage_3),
    .i_busy(i_busy[2]),
    .i_strobe(i_strobe[2]),
    .o_busy(o_busy[2]),
    .o_strobe(o_strobe[2]),
    .in_tlast(i_tlast[2]),
    .out_tlast(o_tlast[2])
  );
  
  //Stage 3
  filter_add # (
    .PIXEL(PIXEL),
    .ROW_LOOP(ROW_LOOP),
    .DATAWIDTH(MEMORYWIDTH)
  )
  filter_add_inst_0 (
    .clk(clk),
    .ARESETN(ARESETN),
    .packed_in_0(top_to_stage_3),
    .packed_in_1(top_to_stage_2),
    .packed_in_2(top_to_stage_1),
    .packed_out_x_0(packed_out_x_0),
    .packed_out_x_2(packed_out_x_2),
    .packed_out_y_0(packed_out_y_0),
    .packed_out_y_2(packed_out_y_2),
    .i_busy(i_busy[3]),
    .i_strobe(i_strobe[3]),
    .o_busy(o_busy[3]),
    .o_strobe(o_strobe[3]),
    .in_tlast(i_tlast[3]),
    .out_tlast(o_tlast[3])
  );

  //stage 4
  filter_sub # (
    .PIXEL(PIXEL),
    .ROW_LOOP(ROW_LOOP),
    .DATAWIDTH(MEMORYWIDTH)
  )
  filter_sub_inst (
    .clk(clk),
    .ARESETN(ARESETN),
    .packed_out_x_0(packed_out_x_0),
    .packed_out_x_2(packed_out_x_2),
    .packed_out_y_0(packed_out_y_0),
    .packed_out_y_2(packed_out_y_2),
    .packed_sum_x(packed_sum_x),
    .packed_sum_y(packed_sum_y),
    .i_busy(i_busy[4]),
    .i_strobe(i_strobe[4]),
    .o_busy(o_busy[4]),
    .o_strobe(o_strobe[4]),
    .in_tlast(i_tlast[4]),
    .out_tlast(o_tlast[4])
  );

  //Stage 5
  filter_sec_add # (
    .PIXEL(PIXEL),
    .ROW_LOOP(ROW_LOOP),
    .DATAWIDTH(MEMORYWIDTH)
  )
  filter_sec_add_inst (
    .clk(clk),
    .ARESETN(ARESETN),
    .packed_sum_x(packed_sum_x),
    .packed_sum_y(packed_sum_y),
    .packed_sum(packed_sum),
    .i_busy(i_busy[5]),
    .i_strobe(i_strobe[5]),
    .o_busy(o_busy[5]),
    .o_strobe(o_strobe[5]),
    .in_tlast(i_tlast[5]),
    .out_tlast(o_tlast[5])
  );

  //Stage 6
  filter_out # (
    .PIXEL(PIXEL),
    .ROW_LOOP(ROW_LOOP),
    .MEMORYWIDTH(MEMORYWIDTH),
    .DATAWIDTH(DATAWIDTH)
  )
  filter_out_inst (
    .clk(clk),
    .ARESETN(ARESETN),
    .packed_sum(packed_sum),
    .M_AXIS_TDATA(M_AXIS_TDATA),
    .i_busy(i_busy[6]),
    .i_strobe(i_strobe[6]),
    .o_busy(o_busy[6]),
    .o_strobe(o_strobe[6]),
    .in_tlast(i_tlast[6]),
    .out_tlast(o_tlast[6])
  );
  
  always_ff @(posedge clk) begin
    if (!ARESETN) begin
      c_state <= INIT;
    end else begin
      c_state <= n_state;
    end
  end

  always_ff @(posedge clk) begin
    if (!ARESETN) begin
      
      RGB0 <= 3'b000;
      RGB1 <= 3'b010;

      S_AXIS_TREADY <= 1'b0;
      shuffle_flag <= 1'b0;
      
      row_top <= 2'b00;
      row_mid <= 2'b01;
      row_end <= 2'b10;
      row_write_ptr <= 'b10;

      col_write_ptr <= 1;
      col_read_ptr <= 0;

      r_en <= 1'b0;
      LED <= 'b0;

    end else begin

      //default behaviors 
      col_read_ptr <= col_read_ptr;
      col_write_ptr <= col_write_ptr;
      S_AXIS_TREADY <= !s_p_busy_o;
      r_en <= 1'b1;
      RGB1 <= 3'b001;

      case (n_state)
        INIT: begin
          RGB0 <= 3'b100;
          
          col_read_ptr <= 0;
          col_write_ptr <= 1;
          
          if (shuffle_flag) begin
            shuffle_flag <= 1'b0;
            row_write_ptr <= (row_write_ptr == 2'b10) ? 2'b00 : (row_write_ptr + 1);
            row_top <= (row_top == 2'b10) ? 2'b00 : (row_top + 1);
            row_mid <= (row_mid == 2'b10) ? 2'b00 : (row_mid + 1);
            row_end <= (row_end == 2'b10) ? 2'b00 : (row_end + 1);
          end
        end

        TRANSMIT: begin
          col_write_ptr <= col_write_ptr + 1;
          col_read_ptr <= col_read_ptr + 1;
        end

        STALL: begin
          r_en <= 1'b0;
          RGB0 <= 3'b010;
        end

        COMPLETE: begin
          S_AXIS_TREADY <= 1'b0;
          shuffle_flag <= 1'b1;
          r_en <= 1'b0;
          RGB0 <= 3'b001;
        end
      endcase
      
      if (S_AXIS_TVALID && S_AXIS_TREADY) begin
        RGB1 <= 3'b100;
        for (int j = 0; j < PIXEL; j++) begin
          bram_in[j] <= S_AXIS_TDATA[(j + 1) * MEMORYWIDTH +: MEMORYWIDTH];
        end
      end
    end
  end

  // state machine 
  always_comb begin
    case (c_state)
      INIT : begin
        if ((S_AXIS_TREADY && S_AXIS_TVALID) && ARESETN) begin
          n_state = TRANSMIT;
        end else begin
          n_state = INIT;
          w_en = 'b0;
        end
      end
      TRANSMIT : begin
        if ((S_AXIS_TREADY && S_AXIS_TVALID)) begin
          w_en = comb_w_en;
          if (S_AXIS_TLAST) begin
            n_state = COMPLETE;
          end else begin
            n_state = TRANSMIT;
          end
        end else begin 
          n_state = STALL;
          w_en = 'b0;
        end
      end
      STALL : begin
        if ((S_AXIS_TREADY && S_AXIS_TVALID)) begin
          w_en = comb_w_en;
          if (S_AXIS_TLAST) begin
            n_state = COMPLETE;
          end else begin
            n_state = TRANSMIT;
          end
        end else begin
          n_state = STALL;
          w_en = comb_w_en;
        end
      end
      COMPLETE : begin
        w_en = 'b0;
        if (M_AXIS_TLAST) begin
          n_state = INIT;
        end else begin
          n_state  = COMPLETE;
        end
      end
      default: n_state = XXX;
    endcase

    case (row_write_ptr)
      2'b00: begin
        comb_w_en = 3'b001;
      end
      2'b01: begin
        comb_w_en = 3'b010;
      end
      2'b10: begin
        comb_w_en = 3'b100;
      end
      default: comb_w_en = 'b0;
    endcase
  end
endmodule



/* 
four states: 
    INIT
      - sets s_axis_tready to 1
      - waits for s_axis_valid
      - if shuffle flag == 1
        - change bram order
        - set shuffle flag to 0
    
    TRANSMIT
      - set strobe signal high
      - increments wptr and rptr
      - stay in state as long as s_rv == 1 && s_l != 1

    STALL
      - if s_rv == 0 stay here
      - set strobe low
    
    COMPLETE
      - stay in state as long as m_l != 1
      - set shuffle flag to 1

  m_axi_valid set by e_p_strobe_o
  !e_p_busy_i set by m_axis_tready
  m_axis_tlast set by e_p_tlast_o

  s_p_tlast_i set by s_axis_tlast
  s_axis_tready set by !s_p_busy_o
  s_p_strobe_i set by s_axis_tvalid

*/

// open_wave_config {/home/lolwut/project/sobel/sobel_refactor/sobel_filter_refactor.wcfg}
// open_wave_config {/home/lolwut/project/sobel/sobel_refactor/sobel_filter_refactor_bram.wcfg}
// open_wave_config {/home/lolwut/project/sobel/sobel_refactor/sobel_filter_refactor_pipeline.wcfg}
// run all
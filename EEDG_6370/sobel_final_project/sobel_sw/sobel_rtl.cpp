#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <unistd.h>
#include <span>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

// Direct Register Mode Register Address Map. We offset by 2 ( foo >> 2) to convert to an integer
// hence, it's 4 byte aligned. Table says 00, 04, etc. but we divide by 4 to get 0, 1, 6, etc.
constexpr int MM2S_DMACR    = 0x00 >> 2; // memory map to streaming dma Control register
constexpr int MM2S_DMASR    = 0x04 >> 2; // memory map to streaming dma Status register
constexpr int MM2S_SA       = 0x18 >> 2; // MM2S Source Address LSB
constexpr int MM2S_SA_MSB   = 0x1C >> 2; // MM2S Source Address MSB (not used)
constexpr int MM2S_LENGTH   = 0x28 >> 2; // MM2S Transfer Length (Bytes)

constexpr int S2MM_DMACR    = 0x30 >> 2; // Streaming to memory map dma Control register
constexpr int S2MM_DMASR    = 0x34 >> 2; // Streaming to memory map dma Status register
constexpr int S2MM_SA       = 0x48 >> 2; // S2MM Source Address LSB
constexpr int S2MM_SA_MSB   = 0x4C >> 2; // S2MM Source Address MSB (not used)
constexpr int S2MM_LENGTH   = 0x58 >> 2; // S2MM Transfer Length (Bytes)

constexpr int DMA_REG_SPACE = 4096 * 16;
constexpr int DMA_0_OFFSET  = 0x41E00000;
constexpr int DMA_1_OFFSET  = 0x41E10000;
// constexpr int DMA_2_OFFSET  = 0x40420000;
// constexpr int DMA_3_OFFSET  = 0x40430000;

constexpr int STATUS_HALTED = 0x01 << 0;
constexpr int CONTROL_RUN   = 0x01 << 0;
constexpr int STATUS_IDLE   = 0x01 << 1;
constexpr int STATUS_INTERR = 0x01 << 4;
constexpr int CONTROL_RESET = 0x01 << 2;
constexpr int EN_ALL_IRQ    = 0x07 << 12;
constexpr int STATUS_IOC    = 0x01 << 12;

constexpr int SOURCE_ADDR   = 0x0E00000;
constexpr int DEST_ADDR     = 0x0F00000;

template <typename R>
void print_mem(std::span<R> myh, const int k, std::string title) {
  std::cout <<std::endl << "Values in "<< title <<" memory location" <<std::endl;
  std::cout << "==============================================" <<std::endl;
  for (int i = 0; i < k / 4; i++)
  {
    if (i % 8 == 0)
    {
    std::cout << std::endl << "Word number: "<< i<<"   ";
    }
    std::cout << std::hex << myh[i] << " ";
  }
  std::cout <<std::endl << "==============================================" <<std::endl;
}

class Dma_Addr_Mmap {
protected:
  void * data_ptr = nullptr;
  int m_fd;
  int m_reg_size;
  int m_offset;

public:
  Dma_Addr_Mmap(const int fd, const int reg_size, const int offset) 
  : m_fd(fd), m_reg_size(reg_size), m_offset(offset) {
    
    data_ptr = mmap(
      nullptr,
      m_reg_size,
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      m_fd,
      m_offset
    );

    if(data_ptr == MAP_FAILED) {
      std::cerr << "mmap failed? You're root. I tihnk." << std::endl;
      throw std::runtime_error ("Nope, next time try sudo.");
    } //Check to debug program
  }

  int getOffset(){
    return m_offset;
  }

  std::span <int> getSpan() {
    return std::span<int> (static_cast<int*>(data_ptr), m_reg_size);
  }

  ~Dma_Addr_Mmap() {
    if(data_ptr) {
      munmap( data_ptr, m_reg_size);
    }
  }
};

class DMA_AXI_Reg : public Dma_Addr_Mmap {
private:
  std::span<int> foo = this->getSpan();
  
  template <typename T>
  void write_reg(std::span<T>& dma_ptr, T dma_reg, T val) {
    dma_ptr[dma_reg] = dma_ptr[dma_reg] ^ val; 
  }
  template <typename T>
  void set_reg_bit(std::span<T>& dma_ptr, T dma_reg, T val) {
    dma_ptr[dma_reg] = dma_ptr[dma_reg] | val; 
  }
  template <typename T>
  void clr_reg_bit(std::span<T>& dma_ptr, T dma_reg, T val) {
    dma_ptr[dma_reg] = dma_ptr[dma_reg] &  ~val; 
  }

  template <typename T>
  int read_reg(std::span<T>& dma_ptr, T dma_reg) {
      return dma_ptr[dma_reg];
  } 
public:
  DMA_AXI_Reg(const int fd, const int reg_size, const int offset) 
  : Dma_Addr_Mmap(fd, reg_size, offset) {}
  ~DMA_AXI_Reg() {}

  void dma_rst(){
    set_reg_bit<int>(foo, MM2S_DMACR, CONTROL_RESET);
    set_reg_bit<int>(foo, S2MM_DMACR, CONTROL_RESET);
  }

  void rtfm(const int source, const int dest, const int elem){
    // dmaDump("reset");
    set_reg_bit<int>(foo, MM2S_DMACR, CONTROL_RUN);
    set_reg_bit<int>(foo, S2MM_DMACR, CONTROL_RUN);
    // dmaDump("enable run");
    setAddr(source, dest);
    dmaRun(elem, elem);

    int cnt = 100;
    while (!(read_reg(foo, S2MM_DMASR) & STATUS_IOC) || (cnt == 0))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        cnt--;
    }
  }

  void dma_irq_clr(){
    set_reg_bit<int>(foo, MM2S_DMASR, STATUS_IOC);
    set_reg_bit<int>(foo, S2MM_DMASR, STATUS_IOC);
    // dmaDump("irq cleared");
  }

  void setAddr(const int source, const int dest) {
    write_reg<int>(foo, MM2S_SA, source);
    write_reg<int>(foo, S2MM_SA, dest);
    // dmaDump("Wrote address");
  }

  void dmaRun(size_t source, size_t dest){
    write_reg<int>(foo, MM2S_LENGTH, source); //source);
    write_reg<int>(foo, S2MM_LENGTH, dest); //dest);
    // dmaDump("wrote length");
  }

  void dmaDump(std::string lol ) {
    std::cout<<std::endl<<"DMA DUMP for "<< lol <<std::endl;
    std::cout << "==============================================" <<std::endl;
    std::cout << "=======     MM2S REGISTERS      ==============" <<std::endl;
    std::cout<< "DMA CTRL REGISTER: "<<std::hex << read_reg(foo, MM2S_DMACR)<<std::endl;
    std::cout<< "DMA STAT REGISTER: "<<std::hex << read_reg(foo, MM2S_DMASR)<<std::endl;
    std::cout<< "  LSB SA REGISTER: "<<std::hex << read_reg(foo, MM2S_SA)<<std::endl;
    std::cout<< "  MSB SA REGISTER: "<<std::hex << read_reg(foo, MM2S_SA_MSB)<<std::endl;
    std::cout<< "  LENGTH REGISTER: "<<std::hex << read_reg(foo, MM2S_LENGTH)<<std::endl<<std::endl;
    std::cout << "==============================================" <<std::endl;
    std::cout << "=======     S2MM REGISTERS      ==============" <<std::endl;
    std::cout<< "DMA CTRL REGISTER: "<< std::hex << read_reg(foo, S2MM_DMACR)<<std::endl;
    std::cout<< "DMA STAT REGISTER: "<< std::hex << read_reg(foo, S2MM_DMASR)<<std::endl;
    std::cout<< "  LSB SA REGISTER: "<< std::hex << read_reg(foo, S2MM_SA)<<std::endl;
    std::cout<< "  MSB SA REGISTER: "<< std::hex << read_reg(foo, S2MM_SA_MSB)<<std::endl;
    std::cout<< "  LENGTH REGISTER: "<< std::hex << read_reg(foo, S2MM_LENGTH)<<std::endl;
    std::cout << "==============================================" <<std::endl;
  }
};

#pragma pack(1)
struct BMPFileHeader {
    uint16_t file_type{0x4D42}; // "BM"
    uint32_t file_size{0};
    uint16_t reserved1{0};
    uint16_t reserved2{0};
    uint32_t offset_data{0};
};

struct BMPInfoHeader {
    uint32_t size{0};
    int32_t width{0};
    int32_t height{0};
    uint16_t planes{1};
    uint16_t bit_count{0};
    uint32_t compression{0};
    uint32_t size_image{0};
    int32_t x_pixels_per_meter{0};
    int32_t y_pixels_per_meter{0};
    uint32_t colors_used{0};
    uint32_t colors_important{0};
};

#pragma pack() // Restore default alignment

int main( int argc, char* argv[]) {
  int ddr_mem_fd;
  int status;
  int control;
  int mnf = DMA_REG_SPACE / 8;
  int file_in_fd;
  int file_out_fd;

  if ( argc < 3)
  {
    std::cerr<<"more aguments plz"<<std::endl;
    throw std::runtime_error("No source or destination value(s) provided.");
  }
  
  std::string in_file = argv[1];
  std::string out_file = argv[2];
  std::cout << "SOBEL Filter"<< std::endl;

  int dma_sel;
  std::string dma_argv = argv[3];
  if (argc > 3)
  {
    if (dma_argv == "1")
    {
      dma_sel = DMA_1_OFFSET;
    }
    else
    {
      dma_sel = DMA_0_OFFSET;
    }
  }
  else
  {
    dma_sel = DMA_0_OFFSET;
  }
  std::cout<< "DMA "<<dma_sel<<" selected"<<std::endl;
  std::cout<< "DMA chosen is '"<<dma_sel<<"'. Argv is '"<<argv[3]<<"'."<<std::endl;


  //Get a file descriptor for /dev/mem/
  if ((ddr_mem_fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1) {
    std::cerr << "Nope, try again. But as root."<<std::endl;
    return 1; 
  } else {
    std::cout << "FD of /dev/mem/ assigned"<<std::endl;
  }

  DMA_AXI_Reg dma_0_virt_addr(ddr_mem_fd, DMA_REG_SPACE, dma_sel); // first dma register

  // DMA_AXI_Reg dma_1_virt_addr(ddr_mem_fd, DMA_REG_SPACE, DMA_1_OFFSET); // second dma register

  Dma_Addr_Mmap virt_source_addr(ddr_mem_fd, DMA_REG_SPACE, SOURCE_ADDR);

  Dma_Addr_Mmap virt_dest_addr(ddr_mem_fd, DMA_REG_SPACE, DEST_ADDR);
  
  BMPFileHeader dest_file_header;
  BMPInfoHeader dest_info_header;
  BMPFileHeader source_file_header;
  BMPInfoHeader source_info_header;
  std::vector<int> pixel_row;

  std::ifstream soruce_file(in_file, std::ios::binary | std::ios::in);
  if (soruce_file.is_open())
  {
    soruce_file.read(reinterpret_cast<char *>( &source_file_header), sizeof(BMPFileHeader));// this line or the previous one is wrong
    soruce_file.read(reinterpret_cast<char * >(&source_info_header), sizeof(BMPInfoHeader));
    // needs to point to the next location.
  }
  else
  {
    throw std::runtime_error ("did not open the source file rite.");
  }

  dest_file_header = source_file_header;
  dest_info_header = source_info_header;
  
  std::ofstream dest_file(out_file, std::ios::binary |std::ios::out);
  if (dest_file.is_open())
  {
    dest_file.write(reinterpret_cast<char *>(&dest_file_header), sizeof(BMPFileHeader));
    dest_file.write(reinterpret_cast<char *> (&dest_info_header), sizeof(BMPInfoHeader));
    std::cout<<"checkpoint 3"<<std::endl;
  }
  else
  {
    throw std::runtime_error("ok so this failed instead?");
  }

  std::cout<<"source info height: " << source_info_header.height<<std::endl;
  std::cout<<"source info width: " << source_info_header.width<<std::endl;
  std::cout<<"dest info height: " << dest_info_header.height<<std::endl;
  std::cout<<"dest info width: " << dest_info_header.width<<std::endl;

  // int pixel = dest_info_header.bit_count/8;
  int pixel_bytes = dest_info_header.bit_count/8;
  char pixel_buffer[4] = {0};
  //*********** meta idea: multiple dma locations, ping pong between them *************** */
  // load first location, start dma, load second location. wait for irq, start dma for 2nd location
  // read third location, write data to first location, clear irq read from 4th, write to 2nd, wait for irq...



  //load in zeros, call reset dma
  memset(virt_source_addr.getSpan().data(), 0, DMA_REG_SPACE / 8);
  // std::cout<<"cleared a span of data"<<std::endl;
  dma_0_virt_addr.dma_rst();
  std::cout<<"reset complete"<<std::endl;
  // std::cout<<"header width: "<< dest_info_header.width<<std::endl;


  dma_0_virt_addr.rtfm(virt_source_addr.getOffset(), virt_dest_addr.getOffset(), dest_info_header.width * 4);
  std::cout<<"dma xfer complete"<<std::endl;

  dma_0_virt_addr.dma_irq_clr();
  // clear interrupt
  for (int i = 0; i < dest_info_header.height; i++)
  {
    for (int j = 0; j < dest_info_header.width; j++)
    {
      // soruce_file.read(reinterpret_cast<char *>(virt_source_addr.getSpan()[j]), pixel);
      soruce_file.read(pixel_buffer, pixel_bytes);
      virt_source_addr.getSpan()[j] = *(reinterpret_cast<int*>(pixel_buffer));
    }
    dma_0_virt_addr.dma_rst();
    dma_0_virt_addr.rtfm(virt_source_addr.getOffset(), virt_dest_addr.getOffset(), dest_info_header.width * 4); 
    
    //wait
    //read contents in mem to file
    for (int k = 0; k < dest_info_header.width; k++)
    {
      // dest_file.write(reinterpret_cast<char *>(virt_dest_addr.getSpan()[k]), pixel);
      *(reinterpret_cast<int*>(pixel_buffer)) = virt_dest_addr.getSpan()[k];
      dest_file.write(pixel_buffer, pixel_bytes);
    }
    dma_0_virt_addr.dma_irq_clr();
    // std::cout<<"line #"<<i<<" of "<<dest_info_header.height<<" done"<<std::endl;
  }
    
  soruce_file.close();
  dest_file.close();

  return 0;
}

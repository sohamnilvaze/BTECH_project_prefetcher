#include "pin.H"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

using namespace std;

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool","o","memtrace.out","specify output file name");

//Global file handle
static ofstream out;

ADDRINT g_mainbase = 0;

VOID ImgLoad(IMG img, VOID* v)
{
    if(IMG_IsMainExecutable(img))
    {
        g_mainbase = IMG_LowAddress(img);
        out << "Main executable loaded at address:" << hex << g_mainbase << "\n";
    }
}

//Record instruction address(PC) for every instruction
VOID RecordInst(ADDRINT ip)
{
    out << "I Ox" << hex << ip << dec << "\n";
}

// Record memory read effective address
VOID RecordMemRead(ADDRINT ip, ADDRINT ea, THREADID tid)
{
    out << "R tid=" << tid << " ip=0x" << std::hex << ip << " ea=0x" << ea << std::dec << "\n";
}

// Record second memory read (some instructions have two memory sources)
VOID RecordMemRead2(ADDRINT ip, ADDRINT ea, THREADID tid)
{
    out << "R2 tid=" << tid << " ip=0x" << std::hex << ip << " ea=0x" << ea << std::dec << "\n";
}

// Record memory write effective address
VOID RecordMemWrite(ADDRINT ip, ADDRINT ea, THREADID tid)
{
    out << "W tid=" << tid << " ip=0x" << std::hex << ip << " ea=0x" << ea << std::dec << "\n";
}

// Instruction instrumentation callback
VOID Instruction(INS ins, VOID *v)
{
    // Instrument every instruction to log its instruction pointer (PC)
    //INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordInst, IARG_INST_PTR, IARG_END);

    // Instrument memory reads (if any)
    //if (INS_IsMemoryRead(ins))
    //{
        // The typical API: IARG_MEMORYREAD_EA gives effective address of the memory read
    //    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead, IARG_INST_PTR, IARG_MEMORYREAD_EA, IARG_THREAD_ID, IARG_END);
    //}

    // Some instructions have two memory read operands
    //if (INS_HasMemoryRead2(ins))
    //{
    //    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead2, IARG_INST_PTR, IARG_MEMORYREAD2_EA, IARG_THREAD_ID, IARG_END);
    //}

    // Instrument memory writes (if any)
    //if (INS_IsMemoryWrite(ins))
    //{
    //    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite, IARG_INST_PTR, IARG_MEMORYWRITE_EA, IARG_THREAD_ID, IARG_END);
    //}
    
    IMG img = IMG_FindByAddress(INS_Address(ins));
    if (!IMG_Valid(img)) return;

    // Only instrument instructions from the main binary
    if (IMG_IsMainExecutable(img)) {
        INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordInst,
                       IARG_INST_PTR, IARG_END);

        if (INS_IsMemoryRead(ins)) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead,
                           IARG_INST_PTR, IARG_MEMORYREAD_EA, IARG_THREAD_ID, IARG_END);
        }
        
        if (INS_HasMemoryRead2(ins))
				{
				    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead2, IARG_INST_PTR, IARG_MEMORYREAD2_EA, IARG_THREAD_ID, IARG_END);
				}
        
        if (INS_IsMemoryWrite(ins)) {
            INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite,
                           IARG_INST_PTR, IARG_MEMORYWRITE_EA, IARG_THREAD_ID, IARG_END);
        }
    }
}

// Fini: called when the program finishes
VOID Fini(INT32 code, VOID *v)
{
    out << "# Program finished\n";
    out.close();
}

// Main
int main(int argc, char *argv[])
{
    // Initialize symbol processing (optional but useful)
    PIN_InitSymbols();

    // Initialize pin
    if (PIN_Init(argc, argv))
    {
        std::cerr << "PIN_Init failed\n";
        return 1;
    }

    out.open(KnobOutputFile.Value().c_str());
    if (!out.is_open())
    {
        std::cerr << "Unable to open output file\n";
        return 1;
    }

    IMG_AddInstrumentFunction(ImgLoad,0);

    // Register Instruction to be called for every instruction
    INS_AddInstrumentFunction(Instruction, 0);

    // Register Fini
    PIN_AddFiniFunction(Fini, 0);

    // Start the program (never returns)
    PIN_StartProgram();

    return 0;
}

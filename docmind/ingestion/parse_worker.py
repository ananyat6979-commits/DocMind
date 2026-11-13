"""Isolated document parser with a 512 MiB address-space/commit limit."""
import json
import os
import sys
from dataclasses import asdict


def bound_resources():
    if os.name != "nt":
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
        resource.setrlimit(resource.RLIMIT_CPU,(45,45))
        return
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_=[("ProcessTime",ctypes.c_longlong),("JobTime",ctypes.c_longlong),
                  ("Flags",wintypes.DWORD),("MinWorkingSet",ctypes.c_size_t),
                  ("MaxWorkingSet",ctypes.c_size_t),("ActiveProcesses",wintypes.DWORD),
                  ("Affinity",ctypes.c_size_t),("Priority",wintypes.DWORD),("Scheduling",wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_=[(name,ctypes.c_ulonglong) for name in ("ReadOps","WriteOps","OtherOps","ReadBytes","WriteBytes","OtherBytes")]
    class Extended(ctypes.Structure):
        _fields_=[("Basic",Basic),("IO",IO),("ProcessMemoryLimit",ctypes.c_size_t),
                  ("JobMemoryLimit",ctypes.c_size_t),("PeakProcessMemory",ctypes.c_size_t),("PeakJobMemory",ctypes.c_size_t)]
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.CreateJobObjectW.restype=wintypes.HANDLE
    kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
    kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    handle=kernel.CreateJobObjectW(None,None)
    if not handle:raise OSError("Cannot create parser resource boundary")
    limits=Extended();limits.Basic.Flags=0x100|0x2
    limits.Basic.ProcessTime=45*10000000
    limits.ProcessMemoryLimit=512*1024**2
    if not kernel.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
        raise OSError("Cannot set parser resource boundary")
    if not kernel.AssignProcessToJobObject(handle,kernel.GetCurrentProcess()):
        raise OSError("Cannot attach parser resource boundary")
    # Keep the handle alive until this worker exits.
    globals()["_JOB_HANDLE"]=handle


def main():
    bound_resources()
    os.environ["DOCMIND_PARSER_WORKER"]="1"
    from docmind.ingestion.loader import load_document
    documents=load_document(sys.argv[1])
    sys.stdout.write(json.dumps([asdict(d) for d in documents],ensure_ascii=True))

if __name__=="__main__":
    try:main()
    except Exception as exc:
        sys.stderr.write(type(exc).__name__+": "+str(exc))
        sys.exit(2)

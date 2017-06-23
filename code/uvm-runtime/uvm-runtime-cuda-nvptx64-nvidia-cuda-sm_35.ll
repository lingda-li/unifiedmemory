; ModuleID = 'uvm-runtime.cu'
source_filename = "uvm-runtime.cu"
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!2}
!nvvm.annotations = !{!3, !4, !3, !5, !5, !5, !5, !6, !6, !5}

!0 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!1 = !{!"clang version 4.0.0 (git@github.com:clang-ykt/clang.git 6b9a2cb4e07445a8cb6c114335887c5ff4efb56d) (git@github.com:clang-ykt/llvm.git 6e30986ed8fb8cb9b32c0b379fe46d17403ec954)"}
!2 = !{i32 1, i32 2}
!3 = !{null, !"align", i32 8}
!4 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!5 = !{null, !"align", i32 16}
!6 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}

; ModuleID = 'uvm-runtime.cu'
source_filename = "uvm-runtime.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.uvmMallocInfo = type { i8*, i64, i8*, i8 }

; Function Attrs: noinline uwtable
define void @_Z9uvmMallocP13uvmMallocInfo(%struct.uvmMallocInfo*) #0 {
  %2 = alloca %struct.uvmMallocInfo*, align 8
  %3 = alloca i64, align 8
  store %struct.uvmMallocInfo* %0, %struct.uvmMallocInfo** %2, align 8
  %4 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %2, align 8
  %5 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %4, i32 0, i32 1
  %6 = load i64, i64* %5, align 8
  store i64 %6, i64* %3, align 8
  %7 = load i64, i64* %3, align 8
  %8 = call noalias i8* @malloc(i64 %7) #3
  %9 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %2, align 8
  %10 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %9, i32 0, i32 2
  store i8* %8, i8** %10, align 8
  %11 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %2, align 8
  %12 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %11, i32 0, i32 0
  %13 = load i64, i64* %3, align 8
  %14 = call i32 @cudaMalloc(i8** %12, i64 %13)
  %15 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %2, align 8
  %16 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %15, i32 0, i32 3
  store i8 0, i8* %16, align 8
  ret void
}

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) #1

declare i32 @cudaMalloc(i8**, i64) #2

; Function Attrs: noinline uwtable
define void @_Z7uvmFreeP13uvmMallocInfo(%struct.uvmMallocInfo*) #0 {
  %2 = alloca %struct.uvmMallocInfo*, align 8
  store %struct.uvmMallocInfo* %0, %struct.uvmMallocInfo** %2, align 8
  %3 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %2, align 8
  %4 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %3, i32 0, i32 0
  %5 = load i8*, i8** %4, align 8
  %6 = call i32 @cudaFree(i8* %5)
  %7 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %2, align 8
  %8 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %7, i32 0, i32 3
  %9 = load i8, i8* %8, align 8
  %10 = trunc i8 %9 to i1
  br i1 %10, label %15, label %11

; <label>:11:                                     ; preds = %1
  %12 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %2, align 8
  %13 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %12, i32 0, i32 2
  %14 = load i8*, i8** %13, align 8
  call void @free(i8* %14) #3
  br label %15

; <label>:15:                                     ; preds = %11, %1
  ret void
}

declare i32 @cudaFree(i8*) #2

; Function Attrs: nounwind
declare void @free(i8*) #1

; Function Attrs: noinline uwtable
define void @_Z9uvmMemcpyP13uvmMallocInfo14cudaMemcpyKind(%struct.uvmMallocInfo*, i32) #0 {
  %3 = alloca %struct.uvmMallocInfo*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i8*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca i64, align 8
  store %struct.uvmMallocInfo* %0, %struct.uvmMallocInfo** %3, align 8
  store i32 %1, i32* %4, align 4
  %8 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %3, align 8
  %9 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %8, i32 0, i32 3
  %10 = load i8, i8* %9, align 8
  %11 = trunc i8 %10 to i1
  br i1 %11, label %12, label %13

; <label>:12:                                     ; preds = %2
  br label %37

; <label>:13:                                     ; preds = %2
  %14 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %3, align 8
  %15 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %14, i32 0, i32 0
  %16 = load i8*, i8** %15, align 8
  store i8* %16, i8** %5, align 8
  %17 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %3, align 8
  %18 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %17, i32 0, i32 2
  %19 = load i8*, i8** %18, align 8
  store i8* %19, i8** %6, align 8
  %20 = load %struct.uvmMallocInfo*, %struct.uvmMallocInfo** %3, align 8
  %21 = getelementptr inbounds %struct.uvmMallocInfo, %struct.uvmMallocInfo* %20, i32 0, i32 1
  %22 = load i64, i64* %21, align 8
  store i64 %22, i64* %7, align 8
  %23 = load i32, i32* %4, align 4
  %24 = icmp eq i32 %23, 1
  br i1 %24, label %25, label %31

; <label>:25:                                     ; preds = %13
  %26 = load i8*, i8** %5, align 8
  %27 = load i8*, i8** %6, align 8
  %28 = load i64, i64* %7, align 8
  %29 = load i32, i32* %4, align 4
  %30 = call i32 @cudaMemcpy(i8* %26, i8* %27, i64 %28, i32 %29)
  br label %37

; <label>:31:                                     ; preds = %13
  %32 = load i8*, i8** %6, align 8
  %33 = load i8*, i8** %5, align 8
  %34 = load i64, i64* %7, align 8
  %35 = load i32, i32* %4, align 4
  %36 = call i32 @cudaMemcpy(i8* %32, i8* %33, i64 %34, i32 %35)
  br label %37

; <label>:37:                                     ; preds = %12, %31, %25
  ret void
}

declare i32 @cudaMemcpy(i8*, i8*, i64, i32) #2

attributes #0 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (git@github.com:clang-ykt/clang.git 6b9a2cb4e07445a8cb6c114335887c5ff4efb56d) (git@github.com:clang-ykt/llvm.git 6e30986ed8fb8cb9b32c0b379fe46d17403ec954)"}

# ============================================
# CUDA C++ 算子编译 Makefile
# ============================================
# 使用方法:
#   make all      - 编译所有算子
#   make gemm     - 编译 GEMM
#   make run      - 编译并运行所有算子
#   make clean    - 清理编译产物
#
# 修改 ARCH 以匹配你的 GPU:
#   RTX 5070/5080/5090: sm_100 (Blackwell)
#   RTX 4090/4080:      sm_89  (Ada Lovelace)
#   RTX 3090/3080:      sm_86  (Ampere)
#   A100:               sm_80  (Ampere)
# ============================================

NVCC = nvcc
ARCH = sm_89
NVCC_FLAGS = -O3 -arch=$(ARCH) --std=c++17

# 所有目标
TARGETS = vector_add gemm softmax rmsnorm fused_add_rmsnorm reduce gemv

.PHONY: all run clean

all: $(TARGETS)

# 编译规则
vector_add: vector_add.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

gemm: gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

softmax: softmax.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

rmsnorm: rmsnorm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

fused_add_rmsnorm: fused_add_rmsnorm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

reduce: reduce.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

gemv: gemv.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# 编译并运行所有
run: all
	@echo "\n========== Vector Add =========="
	./vector_add
	@echo "\n========== Reduce =========="
	./reduce
	@echo "\n========== GEMV =========="
	./gemv
	@echo "\n========== GEMM =========="
	./gemm
	@echo "\n========== Softmax =========="
	./softmax
	@echo "\n========== RMSNorm =========="
	./rmsnorm
	@echo "\n========== Fused Add + RMSNorm =========="
	./fused_add_rmsnorm

clean:
	rm -f $(TARGETS)

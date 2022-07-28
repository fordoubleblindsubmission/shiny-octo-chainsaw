OPT?=fast
VALGRIND?=0
INFO?=0
ENABLE_TRACE_TIMER?=0
CYCLE_TIMER?=0
DEBUG?=0
CILK?=0
SANITIZE?=0
VQSORT?=1

ifeq ($(VQSORT),0)
$(warning sorting and thus batch inserts is much faster with vqsort)
endif


TAPIR_DIR=../OpenCilk-9.0.1-Linux/

CFLAGS := -Wall -Wextra -O$(OPT) -g  -std=c++20 -mcx16 -IParallelTools/ -Itlx/

ifeq ($(SANITIZE),1)
ifeq ($(CILK),1)
CFLAGS += -fsanitize=cilk -fno-omit-frame-pointer
# CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
else
CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
endif
endif

LDFLAGS := -lrt -lm -lpthread -lm -ldl -latomic
ifeq ($(VQSORT),1)
LDFLAGS += -lhwy -lhwy_contrib 
endif
# -ljemalloc

ifeq ($(VALGRIND),0)
CFLAGS += -march=native
endif

DEFINES := -DENABLE_TRACE_TIMER=$(ENABLE_TRACE_TIMER) -DCYCLE_TIMER=$(CYCLE_TIMER) -DCILK=$(CILK) -DDEBUG=$(DEBUG) -DVQSORT=$(VQSORT)

ifeq ($(CILK),1)
CFLAGS += -fopencilk -DPARLAY_CILK
ONE_WORKER = CILK_NWORKERS=1
endif


ifeq ($(DEBUG),0)
CFLAGS += -DNDEBUG
endif


ifeq ($(INFO), 1) 
# CFLAGS +=  -Rpass-missed="(inline|loop*)" 
#CFLAGS += -Rpass="(inline|loop*)" -Rpass-missed="(inline|loop*)" -Rpass-analysis="(inline|loop*)" 
CFLAGS += -Rpass=.* -Rpass-missed=.* -Rpass-analysis=.* 
endif


all: basic test profile opt pma_vs_vector
 
basic: test.cpp PMA.hpp AlignedAllocator.hpp btree.h leaf.hpp CPMA.hpp
	$(CXX) $(CFLAGS) $(DEFINES) $(LDFLAGS) -o $@ test.cpp

pma_vs_vector: pma_vs_vector.cpp CPMA.hpp
	$(CXX) $(CFLAGS) $(DEFINES) $(LDFLAGS) -o $@ pma_vs_vector.cpp

profile: test.cpp PMA.hpp AlignedAllocator.hpp btree.h leaf.hpp CPMA.hpp
	$(CXX) $(CFLAGS) $(DEFINES) -DNDEBUG $(LDFLAGS) -fprofile-instr-generate -o $@ test.cpp
	./profile p 100000
	llvm-profdata-10 merge -output=code.profdata default.profraw
	
test_leaf: basic
	stdbuf --output=L ./basic v_leaf > test_out/v_leaf  || (echo "verification test_leaf failed $$?"; exit 1)
	@echo "test_leaf passed"

test_uncompressed: basic
	stdbuf --output=L ./basic v_uncompressed > test_out/v_uncompressed  || (echo "verification test_uncompressed failed $$?"; exit 1)
	@echo "test_uncompressed passed"

test_compressed: basic
	stdbuf --output=L ./basic v_compressed > test_out/v_compressed  || (echo "verification test_compressed failed $$?"; exit 1)
	@echo "test_compressed passed"

test_batch: basic
	stdbuf --output=L ./basic v_batch > test_out/v_batch || (echo "verification test_batch failed $$?"; exit 1)
	@echo "test_batch passed"

test : test_leaf test_uncompressed test_compressed test_batch
	@echo "all tests done"

cov:
	$(CXX) $(CFLAGS) -fprofile-instr-generate=test_out/cov.%p.profraw -fcoverage-mapping $(DEFINES)  $(LDFLAGS) -o test test.cpp
	@mkdir -p test_out
	./test v || (echo "verification failed $$?"; exit 1)
	llvm-profdata-10 merge -output=test_out/cov.profdata test_out/cov.*.profraw
	llvm-cov-10 show -Xdemangler llvm-cxxfilt-10 -Xdemangler -n ./test -instr-profile=test_out/cov.profdata > coverage_report
	llvm-cov-10 show -format=html -Xdemangler llvm-cxxfilt-10 -Xdemangler -n ./test -instr-profile=test_out/cov.profdata > coverage_report.html
	llvm-cov-10 report ./test -instr-profile=test_out/cov.profdata > coverage_summary

opt: profile
	$(CXX) $(CFLAGS) $(DEFINES) -DNDEBUG $(LDFLAGS) -fprofile-instr-use=code.profdata -o $@ test.cpp

tidy:
	clang-tidy -header-filter=.*  --checks='clang-diagnostic-*,clang-analyzer-*,*,-hicpp-vararg,-cppcoreguidelines-pro-type-vararg,-fuchsia-default-arguments,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-fuchsia-overloaded-operator,-llvm-header-guard,-cppcoreguidelines-owning-memory,-readability-implicit-bool-conversion,-cppcoreguidelines-pro-type-cstyle-cast,-google-readability-casting,-misc-definitions-in-headers,-hicpp-no-malloc,-cppcoreguidelines-no-malloc,-*-use-auto,-readability-else-after-return,-cppcoreguidelines-pro-bounds-constant-array-index,-hicpp-no-array-decay,-cppcoreguidelines-pro-bounds-array-to-pointer-decay'   test.cpp

cachegrind: opt
	valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=cachegrind.log  ./opt p 100000
	cg_annotate --auto=yes cachegrind.log > cachegrind.out
	rm cachegrind.log

# basic: test.o
# 	$(CXX) $(LDFLAGS) -o $@ $^ 

# profile: test_profile.o
# 	$(CXX) $(LDFLAGS) -fprofile-instr-generate -o $@ $^ 
# 	./profile
# 	/home/AUTHOR/OpenCilk-9.0.1-Linux/bin/llvm-profdata merge -output=code.profdata default.profraw

# opt: test_opt.o profile code.profdata
# 	$(CXX) $(LDFLAGS) -fprofile-instr-use=code.profdata -o $@ test_opt.o


clean:
	rm -f *.o opt profile basic code.profdata default.profraw

# test.o:    test.cpp PMA.hpp
# 	$(CXX) $(CFLAGS) -c test.cpp

# test_profile.o:    test.cpp PMA.hpp
# 	$(CXX) $(CFLAGS) -fprofile-instr-generate  -c -o test_profile.o test.cpp

# test_opt.o:    test.cpp PMA.hpp profile
# 	$(CXX) $(CFLAGS) -fprofile-instr-use=code.profdata  -c -o test_opt.o test.cpp

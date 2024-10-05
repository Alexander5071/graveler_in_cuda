COMPILER = nvcc
CFLAGS = -Iinclude -dc -Xcompiler # -Wall
OBJS = graveler.o utils.o
EXEC = graveler.exe
MAX_ROLLS = 1000000000

build: $(EXEC)

$(EXEC): $(OBJS)
	$(COMPILER) $(ARCH) $(OBJS) -o $(EXEC) $(LIBS)

graveler.o: graveler.cu utils.o
	$(COMPILER) $(CFLAGS) -c $< -o $@

utils.o: utils.cu
	$(COMPILER) $(CFLAGS) -c $< -o $@

run: $(EXEC)
	.\$(EXEC) $(MAX_ROLLS)

clean:
	del /q *.o *.exp *.lib $(EXEC) profile.ncu-rep

.PHONY: build run clean

RM = /bin/rm -f
CC = gcc
CFLAGS = -Wall -O3 -fopenmp -ffast-math
LDFLAGS= -lm -fopenmp

OBJS = strassen.o strassen-parallel.o
EXECUTABLE = strassen strassen-parallel

all: $(EXECUTABLE)

#$(EXECUTABLE): $(OBJS)
#	$(CC) -o $@ $(OBJS) $(LDFLAGS)
$(EXECUTABLE): %: %.o
	$(CC) -o $@ $< $(LDFLAGS)
#$(OBJS): %: %.c
#	$(CC) $(CFLAGS) -c $@
strassen.o: strassen.c
	$(CC) $(CFLAGS) -c strassen.c
strassen-parallel.o: strassen-parallel.c
	$(CC) $(CFLAGS) -c strassen-parallel.c
clean:
	$(RM) $(EXECUTABLE) $(OBJS)



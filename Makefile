RM = /bin/rm -f
CC = gcc
CFLAGS = -Wall -pg -O3 -fopenmp -ffast-math -funroll-loops
LDFLAGS= -lm -pg -fopenmp

OBJS = strassen-serial.o strassen-parallel.o
EXECUTABLE = strassen-serial strassen-parallel

all: $(EXECUTABLE)

$(EXECUTABLE): %: %.o
	$(CC) -o $@ $< $(LDFLAGS)
strassen-serial.o: strassen-serial.c
	$(CC) $(CFLAGS) -c strassen-serial.c
strassen-parallel.o: strassen-parallel.c
	$(CC) $(CFLAGS) -c strassen-parallel.c
clean:
	$(RM) $(EXECUTABLE) $(OBJS)



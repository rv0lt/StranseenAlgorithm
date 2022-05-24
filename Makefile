RM = /bin/rm -f
CC = gcc
CFLAGS = -Wall -O3 -fopenmp -ffast-math
LDFLAGS= -lm -fopenmp

OBJS = strassen-serial.o strassen-parallel.o
EXECUTABLE = strassen-serial strassen-parallel

all: $(EXECUTABLE)

#$(EXECUTABLE): $(OBJS)
#	$(CC) -o $@ $(OBJS) $(LDFLAGS)
$(EXECUTABLE): %: %.o
	$(CC) -o $@ $< $(LDFLAGS)
#$(OBJS): %: %.c
#	$(CC) $(CFLAGS) -c $@
strassen-serial.o: strassen-serial.c
	$(CC) $(CFLAGS) -c strassen-serial.c
strassen-parallel.o: strassen-parallel.c
	$(CC) $(CFLAGS) -c strassen-parallel.c
clean:
	$(RM) $(EXECUTABLE) $(OBJS)


